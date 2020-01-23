import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import re
from stl import mesh
from mpl_toolkits import mplot3d
import subprocess
import pickle
#from deap import creator, base

import shutil

################################################################################
# Simple polygonisation
# def. a simple polygon is a non-intersecting polygon.
################################################################################

def is_above_line(point_a, point_b, test_point):
    """
    A method to check whether a point is above the line between two given points
    or not.

    Args:
        point_a (numpy array): a point on the line.
        point_b (numpy array): another point on the line.
        test_point (numpy array): the point that needs to be tested.

    Returns:
        state (bool): whether the proposition is True or False.
    """
    d_1 = (point_b[0] - point_a[0])
    d_2 = (test_point[1] - point_a[1])
    d_3 = (point_b[1] - point_a[1])
    d_4 = (test_point[0] - point_a[0])
    pos = (d_1 * d_2) - (d_3 * d_4)
    if pos >= 0: # this means anything on the line is being considered as above
                 # the line, this is useful in generating simple polygon.
        return True
    else:
        return False

def simple_polygonisation(sample_points):
    """
    A method to rearrange a set of points so that they generate a simple
    polygon. Simple polygons are useful as it is not self-intersecting.

    This is achieved y finding the left most and right most points in the set,
    and then checking which point lie above and below this line. The points
    that lie above the line can be rearranged left to right, and the ones below
    can be rearranged right to left. The sequence of points then generate
    vertices of a simple polygon.

    Args:
        sample_points (numpy array): a set of points.

    Returns:
        seq (numpy array): rearranged points.
    """
    # find the extreme points with respect to x-axis
    left_most = np.argmin(sample_points[:,0])
    right_most = np.argmax(sample_points[:,0])
    # exclude the extreme points from candidates
    candidates = np.delete(sample_points, [left_most, right_most], axis=0)
    # populate seperate sets for points above and below
    set_a, set_b = [], []
    for p in candidates:
        if is_above_line(sample_points[left_most], sample_points[right_most], p):
            set_a.append(p)
        else:
            set_b.append(p)
    set_a, set_b = np.array(set_a), np.array(set_b) # more efficient than
                                                    # np.append
    # generate the final sequence
    seq = np.array([sample_points[left_most]])
    if len(set_a) > 0:
        set_a = set_a[set_a[:,0].argsort()]
        seq = np.concatenate((seq, set_a), axis=0)
    seq = np.concatenate((seq, [sample_points[right_most]]), axis=0)
    if len(set_b) > 0:
        set_b = set_b[(-set_b[:,0]).argsort()]
        seq = np.concatenate((seq, set_b), axis=0)
    seq = np.concatenate((seq, [sample_points[left_most]]), axis=0)
    return seq

def test_simple_polygonisation(n_points=20):
    """
    A method to visually test simple polygonisation. It plots the outcome.

    Args:
        n_points (int): number of points to sort.

    """
    # generate random sample points.
    sample_points = np.random.random_sample((n_points,2))*10
    # generate simple polygon
    seq = simple_polygonisation(sample_points)
    # plot polygon
    plt.figure()
    plt.plot(seq[:,0], seq[:,1], color="blue", marker="s", alpha=0.5)


def rearrange_points(points, order=1):
    '''
    A method to rearrange arrays according to the given order.

    Args:
        points (numpy array): a set of points.

    Kwargs:
        order (int [1,4]): rearranging order. keys::
            1 -- increasing / horizontal
            2 -- decreasing / horizontal
            3 -- increasing / vertical
            4 -- increasing / vertical

    Returns:
        points (numpy array): rearranged points.
    '''
    new_points = points.copy() # this is necessary; otherwise it leads to error
                               # prone code (see numpy documentation)
    if order == 1:
        return new_points[points[:,0].argsort()]
    elif order == 2:
        return new_points[(-1*points[:,0]).argsort()]
    elif order == 3:
        return new_points[points[:,1].argsort()]
    elif order == 4:
        return new_points[(-1*points[:,1]).argsort()]

################################################################################
# Subdivision algorithms.
################################################################################

def catmull(P):
    """
    A method to generate subdivision curves with given control points.

    Args:
        P (numpy array): control points.

    Returns:
        Q (numpy array): generated points on the subdivision curve.
    """
    N = P.shape[0]
    Q = np.zeros((2*N-1, 2), 'd')
    Q[0,:] = P[0,:]
    for i in range(0,N-1):
        if i > 0:
            Q[2*i,:] = (P[i-1,:]+6*P[i,:]+P[i+1,:])/8
        Q[2*i+1,:] = (P[i,:]+P[i+1,:])/2
    Q[-1,:] = P[-1,:]
    return Q

class ControlPolygon2D(object):
    """
    The ControlPolygon2D class defines the control bounadary within which
    any valid curve must reside. The vertices of the control region is accepted
    as inputs, where the sequence specifies the polygon; as such the first and
    the last points must be the same to construct a closed polygon.

    For instance, a polygon with points a, b, c and d may be represented as:
    [a, b, c, d, a]; this must construct a non-intersecting polygon. Note that,
    each point should contain horizontal and vertical coordinates,
    e.g. a = [ax, ay].

    The end points for the subdivision curve should be included in this control
    polygon. The indices of these end points must be specified. For instance,
    if a dn c are the end points for the subdivion curve, then they should be
    specified by setting fixed_points = [[0], [2]]. In the interest of
    smoothness multiple end points may be grouped together (and specified
    through their indicies) so that the curve remains differentiable where it
    leaves the surface of the base shape.

    """

    def __init__(self, vertices, fixed_points, num_points, boundary=None,
                 dim=2):
        '''
        Args:
            vertices (numpy array): vertices of the control region.
            fixed_points (list - 2D): indices of the end points for the
                                        subdivion curves.
            num_points (int): number of control points for any subdivision
                                curve within this polygon.

        Kwargs:
            boundary (numpy array): if a specific rectangular region within
                                    the control polygon is sought wihtin which
                                    any curve will be generated. This should be
                                    structured as follows: [min_x, min_y,
                                    max_x, max_y].
            dim (int): number of dimensions. It should always be 2 in this
                        particular case.

        '''
        self.polygon = Path(vertices)
        self.vertices = self.polygon.vertices
        self.fixed = fixed_points
        self.num_points = num_points
        self.mask = self.__generate_mask()
        self.boundary = boundary
        self.dim = dim

    def __generate_mask(self):
        """
        A (private) method to generate a mask for the control points of the any
        subdivision curves within the polygon. This mask is used to stop
        mutating any of the end points among the control points.

        Returns:
            mask (numpy array): indicates stationary and non-stationary control
                                points.
        """
        mask = np.concatenate([np.ones(len(self.fixed[0])),
                                np.zeros(self.num_points),
                                np.ones(len(self.fixed[1]))])
        return mask

    def random_sample(self):
        '''
        Generate random control points for subdivision curves within the
        specified boundary polygon. Note that any evolutinoary algorithms use
        this method to generate a random individual within a specified control
        polygon.

        Return:
            control_points (numpy array): a set of control points from which a
                                            subdivision curve may be extracted.
        '''
        if self.boundary is None:   # No boundary specified
            min_x, min_y, max_x, max_y = np.min(self.vertices[:,0]),\
                            np.min(self.vertices[:,1]),\
                            np.max(self.vertices[:,0]),\
                            np.max(self.vertices[:,1])
        else:
            min_x, min_y, max_x, max_y = self.boundary
        points = np.zeros((self.num_points, self.dim))
        count = 0
        while count < self.num_points:  # sample until get something valid
            random_point = [np.random.uniform(min_x, max_x), \
                            np.random.uniform(min_y, max_y)]
            if (self.polygon.contains_points([random_point])):
                points[count] = random_point
                count += 1
        #points = rearrange_points(points)
        #return np.vstack([self.vertices[self.fixed[0]], \
        #        points, self.vertices[self.fixed[1]]])

        return self.vector_projection_rearrange(points)

    def vector_projection_rearrange(self, points):
        ab = self.vertices[self.fixed[1][0]] - self.vertices[self.fixed[0][-1]]
        ap = points - self.vertices[self.fixed[0][-1]]
        proj = np.dot(ap, ab.T)/np.linalg.norm(ab)
        #print (proj, np.argsort(proj))
        return np.vstack([self.vertices[self.fixed[0]], \
                        points[np.argsort(proj)], \
                        self.vertices[self.fixed[1]]])

    def get_sequence(self, points):
        '''
        only for 1 control points. Do not use it for anything else.
        '''
        return np.vstack([self.vertices[self.fixed[0]], \
                        points, \
                        self.vertices[self.fixed[1]]])

def calculate_curvature(P):
    """
    A method to calculate the curvature objective function, i.e. the second
    derivative for a given subdivision curve. If there is only one curve then the
    total curvature is the sum of squared second derivative. On the other hand,
    if there are multiple control polygons then the maximum curvature is returned.

    Args:
        P (numpy array): all points generated through catmull subdivision.
    """
    y = P[:,1].copy()
    x = P[:,0].copy()
    dx = np.gradient(x)
    yd = np.gradient(y, dx)
    ydd = np.gradient(yd, dx)
    return np.sum(ydd**2)

################################################################################
# supporting routines.
################################################################################

def get_words(text):
    """
    A method to separate the words in a string.

    Args:
        text (str): input text string.

    Return:
        words (list): a list of words in the string.
    """
    return re.compile('\w+').findall(text)

################################################################################
# STL routines.
################################################################################

def subdc_to_stl_2d(control_points, n_iter, thickness=np.array([0,0,0.1]),
                     file_directory=None, file_name=None, draw=False):
    """
    A method to save a subdivision curve as a STL file.

    Args:
        control_points (numpy array): 2D array of control points fro
                                        subdivision curves; each row is
                                        a particular control point.
        n_iter (int): Number of subdivision iterations. Higher number will
                        produce a smoother curve.

    Kwargs:
        thickness (numpy array): The thickness along z-axis (last element in the
                                    array). It can also be used to offset
                                    x- and y-axis.
        file_directory (str): The destination directory for the STL file.
        file_name (str): The name of the STL file.
        draw (bool): Control for drawing graphs.


    """
    data_points = catmull(control_points)
    for i in range(n_iter):
        data_points = catmull(data_points)
    # add z-axis
    data_points = np.concatenate((data_points, \
                    np.zeros((len(data_points),1))-(thickness[-1]/2.0)), axis=1)
    # calculate total number of faces
    num_faces = len(data_points)*2
    data = np.zeros(num_faces, dtype=mesh.Mesh.dtype)
    count = 0
    # generate the face seqeunce for STL
    for i in range(0, num_faces, 2):
        next_count = count + 1
        if next_count == len(data_points):
            break
        data['vectors'][i] = np.array([data_points[count],
                                       data_points[next_count],
                                       data_points[count]+thickness])
        data['vectors'][i+1] = np.array([data_points[count]+thickness,
                                       data_points[next_count]+thickness,
                                       data_points[next_count]])
        count += 1
    # generate mesh
    m = mesh.Mesh(data.copy())
    # debug
    if draw:
        draw_stl_from_mesh(m)
    # save file
    if file_directory is None:
        if file_name is None:
            m.save('test.stl')
        else:
            m.save(file_name)
    else:
        m.save(file_directory + file_name)

def subdc_to_stl_mult(individuals, n_iter, thickness=np.array([0,0,0.1]),
                        file_directory=None, file_name=None, draw=False):
    """
    A method to save multiple subdivision curves in a single STL file. This may be
    used for multiple control regions.

    Args:
        individuals (numpy array): DEAP control point individuals.
        n_iter (int): Number of subdivision iterations. Higher number will
                        produce a smoother curve.

    Kwargs:
        thickness (numpy array): The thickness along z-axis (last element in the
                                    array). It can also be used to offset
                                    x- and y-axis.
        file_directory (str): The destination directory for the STL file.
        file_name (str): The name of the STL file.
        draw (bool): Control for drawing graphs.
    """
    all_data = []
    curv_data = []
    # generate faces for each individual curve in the DEAP inidividuals
    for control_points in individuals:
        data_points = catmull(control_points)
        for i in range(n_iter):
            data_points = catmull(data_points)
        curv_data.append(data_points.copy())
        #curv_data.append(calculate_curvature(data_points)) # who did this?
        data_points = np.concatenate((data_points, np.zeros((len(data_points),1))-(thickness[-1]/2.0)), axis=1)
        num_faces = len(data_points)*2

        data = np.zeros(num_faces, dtype=mesh.Mesh.dtype)
        count = 0
        for i in range(0,num_faces,2):
            next_count = count + 1
            if next_count == len(data_points):
                break
            data['vectors'][i] = np.array([data_points[count],
                                           data_points[next_count],
                                           data_points[count]+thickness])
            data['vectors'][i+1] = np.array([data_points[count]+thickness,
                                           data_points[next_count]+thickness,
                                           data_points[next_count]])
            count += 1
        # combine faces
        all_data.append(data)
    all_data = np.concatenate(all_data)
    # generate mesh
    m = mesh.Mesh(all_data)
    # debug
    if draw:
        draw_stl_from_mesh(m)
    if file_directory is None:
        if file_name is None:
            m.save('test.stl')
        else:
            m.save(file_name)
    else:
        m.save(file_directory + file_name)
    return np.array(curv_data)

def draw_stl_from_file(file_name):
    """
    A method to draw a STL mesh from a given file.

    Args:
        file_name (str): name of the file.
    """
    plt.ion()
    m = mesh.Mesh.from_file(file_name)
    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    # Render the cube faces
    #for m in meshes:
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
    # Auto scale to the mesh size
    scale = m.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

def draw_stl_from_mesh(m):
    """
    A method to draw numpy-stl mesh.

    Args:
        m (numpy-stl mesh): mesh object.

    """
    plt.ion()
    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    # Render the cube faces
    #for m in meshes:
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))

    # Auto scale to the mesh size
    scale = m.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)


def circle(r, h, k, theta):
    x = r * np.cos(theta) + k
    y =  r * np.sin(theta) + h
    return [x, y]

def circle_to_stl(rs, xs, ys, thickness=np.array([0,0,1]), c_res=1000, \
                        file_directory=None, file_name=None, draw=False):
    all_data = []
    for i in range(len(rs)):
        r = rs[i]
        xcen = xs[i]
        ycen = ys[i]
        # generate the points at the perimeter
        data_points = circle(r, ycen, xcen, np.linspace(0, 2*np.pi, c_res))
        data_points = np.rot90(data_points)
        # add z-axis
        data_points = np.concatenate((data_points, \
                    np.zeros((len(data_points),1))-(thickness[-1]/2.0)), axis=1)
        # calculate total number of faces
        num_faces = len(data_points)*2
        data = np.zeros(num_faces, dtype=mesh.Mesh.dtype)
        count = 0
        # generate the face seqeunce for STL
        for i in range(0, num_faces, 2):
            next_count = count + 1
            if next_count == len(data_points):
                break
            data['vectors'][i] = np.array([data_points[count],
                                           data_points[next_count],
                                           data_points[count]+thickness])
            data['vectors'][i+1] = np.array([data_points[count]+thickness,
                                           data_points[next_count]+thickness,
                                           data_points[next_count]])
            count += 1
        all_data.append(data)
    all_data = np.concatenate(all_data)
    # generate mesh
    m = mesh.Mesh(all_data.copy())
    if draw:
        draw_stl_from_mesh(m)
    # save file
    if file_directory is None:
        if file_name is None:
            m.save('ribbon.stl')
        else:
            m.save(file_name)
    else:
        m.save(file_directory + file_name)


################################################################################
# file operations
################################################################################

def RemoveCase(dirc):
    """
    A method to remove a particular case directory.

    Args:
        dirc (str): case directory.
    """
    if os.path.exists(dirc):
        shutil.rmtree(dirc)
    #subprocess.call(['rm', '-r', dirc])

def RestoreCase(dirc, dest):
    """
    A method to restrore case from a bakcup directory.

    Args:
        dirc (str): backup directory including the case name.
        dest (str): destination directory.
    """
    subprocess.call(['cp', '-r', dirc, dest])

def read_data(filename, data_start_format, data_end_format,
                exclude=['(',')'], type_id = "object", data_type=float):
    """
    A method to read the mesh centre coordinates from OpenFoam files. This may
    be used to read other files as well.

    Args:
        filename (str): file name with directory.
        data_start_format (str): the starting line from wehre data shoiuld be read.
        data_end_format (str): the line before which it should stop.

    Kwargs:
        exclude (list): list of characters that should be excluded from the data.
        type_id (str): the type identifier of the data in OpenFoam file.
        data_type (str): the data type being read.

    Returns:
        file_type (str): the type identifier of the extracted data.
        data (numpy array): the extracted data.

    """
    data = []
    with open(filename, 'r') as data_file:
        grab_lines = False
        for line in data_file:
            if grab_lines and len([i for i in exclude if i in line])==0:
                data.append(data_type(line))
            if line.startswith(data_start_format):
                grab_lines = True
            elif grab_lines and (data_end_format in line):
                grab_lines = False
            elif type_id in line:
                file_type = get_words(line)
    return file_type, data

def write_data(filename, data, data_start_format, data_end_format,
                exclude=['(',')']):
    """
    A method to replace data in an OpenFoam files.

    Args:
        filename (str): file name with directory.
        data_start_format (str): the starting line from wehre data should be
                                    written.
        data_end_format (str): the line before which it should stop.

    Kwargs:
        exclude (list): list of characters that should be excluded from the data.

    """
    with open(filename, 'r') as data_file:
        file_data = data_file.readlines()
    write_lines = False
    count = 0
    data = [len(data)] + data
    for ind in range(len(file_data)):
        if data_start_format in file_data[ind]:
            write_lines = True
            ind += 1
        elif write_lines and len([i for i in exclude if i in file_data[ind]])==0:
            file_data[ind] = str(data[count]) + "\n"
            count += 1
        elif write_lines and (data_end_format in file_data[ind]):
            write_lines = False
            break
    with open(filename, 'w') as data_file:
        data_file.writelines(file_data)
    print ("File updated: ", filename)

def write_data_to_csv(filename, data):
    """
    A method to write data in a csv file.

    Args:
        filename (str): file name with directory.
        data (list or numpy array): data to be written.

    """
    with open(filename, "wb") as csvfile:
        data_writer = csv.writer(csvfile, delimiter=",", quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
        for i in data:
            data_writer.writerow(i)


def pickle_data(filename, data):
    """
    A method to conduct object serialisation.

    Args:
        filename (str): destination file name.
        data (obj): python object.

    """
    f = open(filename, "wb")
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()

# def unpickle_data(filename):
    # """
    # A method to load serialised (pickled) object from file.

    # What is the purpose of this?

    # Args:
        # filename (str): input file name.
    # """
    # creator.create('Fitness', base.Fitness, weights=(1.0, -1.0))
    # creator.create('Individual', list, fitness=creator.Fitness)
    # f = open(filename, "rb")
    # data = pickle.load(f)
    # f.close()
    # return data

################################################################################
# Deap plotting
################################################################################

def deap_plot_2D_front(population, xlab="$f_1$", ylab="$f_2$",
                        colour="blue"):
    """
    A method to visualise 2D front.

    Args:
        population (list): list of individuals on the estimated front.

    Kwargs:
        xlab (str): horizontal axis label.
        ylab (str): vertical axis label.
        colour (str): required colour (make sure the colour exist in
                        matplotlib).
    """
    plt.ion()
    # extract estimated front
    est_front_x, est_front_y = [i.fitness.values[0] for i in population], \
                                   [i.fitness.values[1] for i in population]
    plt.figure()
    plt.scatter(est_front_x, est_front_y,
                color=colour, label=("Estimated Front"))
    plt.xlabel(xlab)
    plt.ylabel(ylab)

def deap_plot_hyp(stats, colour="blue"):
    """
    A method to plot the hypervolume progression using DEAP. It uses the
    logs accumulated during evolution in DEAP.

    Args:
        stats (deap log): statistics accumulated during evolution.

    Kwargs:
        colour (str): required colour (make sure the colour exists in
                        matplotlib)
    """
    plt.ion()
    # plot hypervolumes
    hyp = []
    for gen in stats:
          hyp.append(gen['hypervolume'])
    plt.figure()
    plt.plot(hyp, color=colour)
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")

################################################################################
# silencing std output
################################################################################

import contextlib
from contextlib import redirect_stdout, contextmanager
import io
#import cStringIO, sys
@contextlib.contextmanager
def nostdout():
    """
    A method to stop standard output to shell.
    """
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            yield
        except Exception as err:
            raise err
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        except Exception as err:
            raise err
        finally:
            sys.stdout = old_stdout
"""

@contextlib.contextmanager
def nostdout():
    '''
    Prevent print to stdout, but if there was an error then catch it and
    print the output before raising the error.
    link: https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
    '''
    saved_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    try:
        yield
    except Exception:
        saved_output = sys.stdout
        sys.stdout = saved_stdout
        print (saved_output.getvalue())
        raise
    sys.stdout = saved_stdout
"""
def plot_solutions(datafile, xind='arr_0', yind='arr_2', init_samples=190):
    res = np.load(datafile)
    X = res[xind]
    Y = res[yind]
    plt.figure()
    plt.scatter(Y[:init_samples,0], Y[:init_samples,1], \
                    marker="s", alpha=0.25, color="black")
    c = np.arange(1, Y[init_samples:].shape[0]+1, 1)
    plt.scatter(Y[init_samples:,0], Y[init_samples:,1], c=c, alpha=0.75)
    cb = plt.colorbar(orientation="horizontal")
    cb.set_label('Sample Number')
    plt.xlabel("Temparature Difference")
    plt.ylabel('Pressure Difference')
    ninds = np.where(CS.nond_ind(Y, [-1]*Y.shape[1])==0)[0]
    plt.scatter(Y[ninds,0], Y[ninds,1], edgecolor="red",facecolor="none",s=80, alpha=0.75)


################################################################################
# Parametric functions
################################################################################
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev as C
from scipy.stats import beta as B
from scipy.stats import dirichlet as D
from scipy.stats import truncnorm as tnorm
from scipy.spatial.distance import euclidean

EXT_BETA_CDF = lambda x, a, b : np.array([B.cdf(x, a[i],b[i]) for i in range(len(a))])

class Chebyshev(object):

    def __init__(self, n_coeffs, coeffs=None, clb=-1, cub=1, domain=[0,1], window=[-1,1]):
        self.nc = n_coeffs
        self.domain = domain
        self.window = window
        self.clb = clb
        self.cub = cub
        # determine ymin
        cf = np.array([self.clb]*n_coeffs)
        self.coeffs = cf
        self.update_function()
        self.ymin = self.function(1)
        # determine ymax
        cf = np.array([self.cub]*n_coeffs)
        self.coeffs = cf
        self.update_function()
        self.ymax = self.function(1)
        # set coefficients
        if coeffs is None:
            self.coeffs = self.random_sample()
        else:
            self.coeffs = coeffs
        self.update_function()

    def update_function(self):
        assert self.nc == len(self.coeffs)
        self.function = C(self.coeffs, domain=self.domain, window=self.window)

    def random_sample(self):
        return np.random.random_sample(self.nc) * (self.cub - self.clb) + self.clb

    def evaluate(self, n_samples, lb, ub):
        # find positions
        positions = (np.arange(n_samples) + 1)/(n_samples + 1)
        # compute function values
        y = self.function(positions)
        # scale between 0 and 1
        y = (y - self.ymin)/(self.ymax - self.ymin)
        # scale between lb and ub
        y = ((ub - lb) * y) + lb
        return y

class MonotonicBetaCDF(object):

    def __init__(self, n_betas, alphas=None, betas=None, omegas=None, \
                            alb=0, aub=5, blb=0, bub=5, wub=1, wlb=0):
        self.nb = n_betas
        self.alb = alb
        self.aub = aub
        self.blb = blb
        self.bub = bub
        self.wub = wub
        self.wlb = wlb
        if alphas is not None:
            self.alphas = alphas
            self.betas = betas
            self.omegas = omegas
        else:
            self.alphas, self.betas, self.omegas = self.random_sample()

    def set_alphas(self, alphas):
        self.alphas = alphas

    def set_betas(self, betas):
        self.betas = betas

    def set_omegas(self, omegas):
        self.org_omegas = omegas
        self.omegas = omegas/np.sum(omegas)

    def random_sample(self):
        alphas = np.random.random_sample(self.nb) * (self.aub - self.alb) + self.alb
        betas = np.random.random_sample(self.nb) * (self.bub - self.blb) + self.blb
        omegas = D.rvs([1]*self.nb)[0]
        return alphas, betas, omegas

    def evaluate(self, n_samples, lb, ub):
        positions = (np.arange(n_samples) + 1)/(n_samples + 1)
        return np.dot(self.omegas, EXT_BETA_CDF(positions, self.alphas, self.betas))\
                    *(ub-lb) + lb

class PipeRow(object):

    def __init__(self, vert_origin, vert_pos, n_pipes, xlb, xub, rlb, rub, \
                 n_coeffs, n_betas,\
                 coeffs=None, clb=-1, cub=1, domain=[0,1], window=[-1,1], \
                 alphas=None, betas=None, omegas=None, \
                 alb=0, aub=10, blb=0, bub=10, radii=None, centers=None, repeat=False):
        self.y0 = vert_origin
        self.y = vert_pos
        self.n_pipes = n_pipes
        self.repeat = repeat
        self.xlb = xlb
        self.xub = xub
        self.rlb = rlb
        self.rub = rub
        if radii is None:
            self.radii = Chebyshev(n_coeffs, coeffs=coeffs, clb=clb, cub=cub,\
                                        domain=domain, window=window)
        else:
            self.radii = radii
        if centers is None:
            self.centers = MonotonicBetaCDF(n_betas, alphas=alphas, betas=betas, \
                                    omegas=omegas, alb=alb, aub=aub, blb=blb, \
                                    bub=bub)
        else:
            self.centers = centers
        self.evaluate()

    def evaluate(self):
        self.x = self.centers.evaluate(self.n_pipes, self.xlb, self.xub)
        self.r = self.radii.evaluate(self.n_pipes, self.rlb, self.rub)

    def draw(self, plt, other=None):
        plt.figure()
        ax = plt.axes()
        for i in range(len(self.r)):
            c = plt.Circle((self.x[i], self.y), self.r[i])
            ax.add_artist(c)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        if other is not None:
            for i in range(len(other.r)):
                c = plt.Circle((other.x[i], other.y), other.r[i])
                ax.add_artist(c)

    def check_constraints(self, other=None):
        if other is None:
            is_valid = True
            for i in range(self.n_pipes-1):
                if self.r[i] + self.r[i+1] >= \
                    euclidean([self.x[i], self.y], [self.x[i+1], self.y]):
                    self.is_valid = False
                    return False
            if self.repeat and np.any(np.array(self.r)>=(self.y - self.y0)/2):
                return False
            return True
        else:
            for i in range(self.n_pipes):
                for j in range(other.n_pipes):
                    if self.r[i] + other.r[j] >= \
                        euclidean([self.x[i], self.y], [other.x[j], other.y]):
                        self.is_valid = False
                        return False
            return True
