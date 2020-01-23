import numpy as np
try:
    from base_class import Interface
except:
    from .base_class import Interface
try:
    from data.support import Chebyshev, MonotonicBetaCDF, PipeRow
except:
    from .data.support import Chebyshev, MonotonicBetaCDF, PipeRow

class ControlPolygonInterface(Interface):
    """
    This is the T junction interface. It helps convert a decision vector to
    shape parameters and vice versa. Most importantly, it help runniig the CFD
    simulation and return function values.
    """

    def __init__(self, control_polygons):
        """
        Parameters.
        -----------
        control_polygons (list of ControlPolygon2D): control polygons that must
                                contain the splines.
        problem (TJunction): a CFD problem.
        """
        self.ncpoly = len(control_polygons)
        self.cpolys = control_polygons

    def constraint(self, x):
        """
        A constraint to check if the decision vector lies within the predefined
        control polygon.

        Returns whether the constraint was violated or not.
        """
        pts_comp = []
        ind = 0
        for i in range(self.ncpoly):
            leng = self.cpolys[i].num_points
            pos = np.reshape(x[ind:ind+(leng*2)], (-1, 2))
            pts_comp.extend([self.cpolys[i].polygon.contains_points([pts]) for pts in pos])
            ind += (leng*2)
        if np.all(pts_comp):
            return True
        return False

    def get_decision_boundary(self):
        """
        The decision vector lies within a hyperrectangle feasible space.

        Returns the lower and upper bounds of that hyper-rectangle.
        """
        lower, upper = [], []
        for i in range(self.ncpoly):
            lb_cand = np.tile(np.min(self.cpolys[i].vertices, axis=0), self.cpolys[i].num_points)
            lower.extend(lb_cand)
            ub_cand = np.tile(np.max(self.cpolys[i].vertices, axis=0), self.cpolys[i].num_points)
            upper.extend(ub_cand)
        return np.array(lower), np.array(upper)

    def convert_decision_to_shape(self, x):
        """
        Convert a decision vector to shape parameters.

        Parameters.
        -----------
        x (numpy array): decision vector.

        Returns shape parameter list.
        """
        assert self.constraint(x), "The decision vector violates constraints."
        splines = []
        ind = 0
        for i in range(self.ncpoly):
            leng = self.cpolys[i].num_points
            mod_x = np.reshape(x[ind:ind+(leng*2)], (-1, 2))
            splines.append(self.cpolys[i].vector_projection_rearrange(mod_x))
            ind += (leng*2)
        return splines

    def convert_shape_to_decision(self, params):
        """
        Convert shape parameters to decision vector.

        Parameters.
        -----------
        params (list of arrays): splines.

        Returns decision vector.
        """
        x = []
        splines = params
        for i in range(self.ncpoly):
            x.extend(np.concatenate(splines[i][len(self.cpolys[i].fixed[0]):self.cpolys[i].num_points+len(self.cpolys[i].fixed[0])]))
        return np.array(x)



class PipeInterface(Interface):

    def __init__(self, vert_origin, vert_positions, xlb, xub, rlb, rub, nlb, nub,\
                        n_coeffs_radii, n_coeffs_num, n_betas,\
                        clb=-1, cub=1, domain=[0,1], window=[-1,1]):
        self.vert_origin = vert_origin # where is the vertical origin
        self.vert_positions =  vert_positions # coordinates of the vertical axis
        self.xlb = xlb # horizontal limits
        self.xub = xub
        self.rlb = rlb # radius limits
        self.rub = rub
        self.nlb = nlb # number of pipes limits
        self.nub = nub
        self.n_coeffs_radii = n_coeffs_radii # coefficient for radii
        self.n_coeffs_num = n_coeffs_num # coefficients for number of pipes
        self.n_betas = n_betas # number of beta functions -- used for
        self.n_pipes_count = Chebyshev(self.n_coeffs_num, clb=clb, cub=cub,\
                                        domain=domain, window=window)
        self.n_pipes = np.rint(self.n_pipes_count.evaluate(len(self.vert_positions), self.nlb, self.nub)).astype('int')
        rows = []
        for i in range(len(self.vert_positions)):
            rows.append(PipeRow(vert_origin, vert_positions[i], self.n_pipes[i], xlb, xub,\
                            rlb, rub, n_coeffs_radii[i], n_betas[i]))

        self.rows = rows
        self.lb, self.ub = self.get_decision_boundary()

    def convert_shape_to_decision(self):
        d = [self.n_pipes_count.coeffs]
        for row in self.rows:
            d.append(row.centers.alphas)
            d.append(row.centers.betas)
            d.append(row.centers.omegas)
            d.append(row.radii.coeffs)
        d = np.concatenate(d)
        assert d.shape[0] == self.lb.shape[0] == self.ub.shape[0]
        return d

    def get_decision_boundary(self):
        lb = [self.n_pipes_count.clb]*len(self.n_pipes_count.coeffs)
        ub = [self.n_pipes_count.cub]*len(self.n_pipes_count.coeffs)
        for row in self.rows:
            lb.extend([row.centers.alb]*len(row.centers.alphas))
            ub.extend([row.centers.aub]*len(row.centers.alphas))
            lb.extend([row.centers.blb]*len(row.centers.betas))
            ub.extend([row.centers.bub]*len(row.centers.betas))
            lb.extend([row.centers.wlb]*len(row.centers.omegas))
            ub.extend([row.centers.wub]*len(row.centers.omegas))
            lb.extend([row.radii.clb]*len(row.radii.coeffs))
            ub.extend([row.radii.cub]*len(row.radii.coeffs))
        return np.array(lb), np.array(ub)

    def update_layout(self, d):
        """
        d = decision variables
        """
        init = 0
        next_ind = self.n_pipes_count.coeffs.shape[0]
        #print("number of pipes coeffcients: ", self.n_pipes_c.coeffs, d[init:next_ind])
        self.n_pipes_count.coeffs = d[init:next_ind]
        self.n_pipes_count.update_function()
        #print(self.n_pipes_count.evaluate(len(self.vert_positions), self.nlb, self.nub), self.nlb, self.nub)
        self.n_pipes = np.rint(self.n_pipes_count.evaluate(len(self.vert_positions), self.nlb, self.nub)).astype('int')
        for i in range(len(self.rows)):
            self.rows[i].n_pipes = self.n_pipes[i]
            #print("row: ", i)
            init += next_ind
            next_ind = self.rows[i].centers.alphas.shape[0]
            #print("number of center alphas: ", self.rows[i].centers.alphas, d[init:init+next_ind])
            self.rows[i].centers.alphas = d[init:init+next_ind]
            init += next_ind
            next_ind = self.rows[i].centers.betas.shape[0]
            #print("number of centter betas: ", self.rows[i].centers.betas, d[init:init+next_ind])
            self.rows[i].centers.betas = d[init:init+next_ind]
            init += next_ind
            next_ind = self.rows[i].centers.omegas.shape[0]
            #self.rows[i].centers.omegas = d[init:init+next_ind]
            omegas = d[init:init+next_ind]
            #print("number of omegas: ", self.rows[i].centers.omegas, omegas)
            self.rows[i].centers.set_omegas(omegas)
            init += next_ind
            next_ind = self.rows[i].radii.coeffs.shape[0]
            #print("number of radii coeffcients: ", self.rows[i].radii.coeffs, d[init:init+next_ind])
            self.rows[i].radii.coeffs = d[init:init+next_ind]
            self.rows[i].radii.update_function()
            self.rows[i].evaluate()

    def constraint(self, d):
        self.update_layout(d)
        for row in self.rows:
            if not row.check_constraints():
                return False
        nrows = len(self.rows)
        for  i in range(nrows):
            for j in range(i+1, nrows, 1):
                if not self.rows[i].check_constraints(other=self.rows[j]):
                    return False
        return True

    def convert_decision_to_shape(self, d):
        self.update_layout(d)
        xs, ys, rs = [], [], []
        for row in self.rows:
            ys.extend( [row.y] * row.n_pipes)
            xs.extend( list(row.x))
            rs.extend( list(row.r))
            if row.repeat:
                ys.extend([(2*row.y0) - row.y]*row.n_pipes)
                xs.extend(xs)
                rs.extend(rs)
        return xs, ys, rs

    def plot(self, fignum=None):
        import matplotlib.pyplot as plt
        plt.ion()
        try:
            plt.figure(fignum)
        except:
            plt.figure()
        ax = plt.gca()
        ax.cla()
        min_x, min_y, max_x, max_y, max_r = 0, 0, 0, 0, 0
        for row in self.rows:
            ys = [row.y] * row.n_pipes
            xs = list(row.x)
            rs = list(row.r)
            if row.repeat:
                ys.extend([(2*row.y0) - row.y]*row.n_pipes)
                xs.extend(xs)
                rs.extend(rs)
            for i in range(len(xs)):
                circle = plt.Circle((xs[i], ys[i]), rs[i], facecolor="blue", edgecolor="black", alpha=0.25)
                ax.add_artist(circle)
            min_x = np.min(np.concatenate([[min_x], xs]))
            max_x = np.max(np.concatenate([[max_x], xs]))
            min_y = np.min(np.concatenate([[min_y], ys]))
            max_y = np.max(np.concatenate([[max_y], ys]))
            max_r = np.max(np.concatenate([[max_r], rs]))
        plt.xlim(min_x - max_r, max_x + max_r)
        plt.ylim(min_y - max_r, max_y + max_r)
        plt.draw()

################################################################################
#tests
################################################################################

def test_control_polygon_interface():
    import numpy as np
    import data.support as support
    import matplotlib.pyplot as plt
    plt.ion()
    n_control = 5
    boundary_file = 'data/PitzDaily/boundary.csv'
    fixed_points_file = 'data/PitzDaily/fixed.csv'
    pts = np.loadtxt(boundary_file, delimiter=',')
    fixed_points = np.loadtxt(fixed_points_file, delimiter=',').astype(int)
    fixed_points = list(fixed_points)
    cpoly = support.ControlPolygon2D(pts, fixed_points, n_control)
    cpolys = [cpoly]
    interface = ControlPolygonInterface(cpolys)
    # visual test with random decision vectors.
    lb, ub = interface.get_decision_boundary()
    xrand = np.random.random((100, 10)) * (ub -lb) + lb
    s = []
    for i in x:
        try:
            s.append(interface.convert_decision_to_shape(i))
        except:
            pass
    for i in s:
        plt.plot(i[0][:,0], i[0][:,1], marker='o', alpha=0.25, lw=3)
        plt.draw()

def test_pipes():
    D = 0.2
    vert_origin = 0
    n_rows = 3
    vert_positions = np.array([-D, 0, D])
    xlb, xub = -D, 3.25*D
    rlb, rub = 0.005, 0.5*D
    nlb, nub = 1, 5
    n_coeffs_radii = [3]*n_rows
    n_coeffs_num = 4
    n_betas = [3]*n_rows
    pipes = PipeInterface(vert_origin, vert_positions, xlb, xub, rlb, rub, \
                            nlb, nub, n_coeffs_radii, n_coeffs_num, n_betas)
    lb, ub = pipes.lb, pipes.ub
    import numpy as np
    for i in range(100):
        d = np.random.random(len(lb)) * (ub - lb) + lb
        d[0] = -1
        d[1:4] = 1
        if pipes.constraint(d):
            print(d)
            pipes.plot(1)
        else:
            print('invalid')
        input()
