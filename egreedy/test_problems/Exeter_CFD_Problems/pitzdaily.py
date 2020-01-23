try:
    from data.SnappyHexOptimise import BasicPitzDailyRun
except:
    from .data.SnappyHexOptimise import BasicPitzDailyRun
try:
    from interfaces import ControlPolygonInterface
except:
    from .interfaces import ControlPolygonInterface
try:
    from base_class import Problem
except:
    from .base_class import Problem
try:
    from data import support #import data.support as support
except:
    from .data import support #as support
import numpy as np

class PitzDaily(Problem, ControlPolygonInterface):

    def __init__(self, settings):
        self.source_case = settings.get('source_case', 'data/PitzDaily/case_fine')
        self.case_path = settings.get('case_path', 'data/PitzDaily/case_single/')
        self.domain_files = settings.get('boundary_files', ['data/PitzDaily/boundary.csv'])
        self.fixed_points_files = settings.get('fixed_points_files', ['data/PitzDaily/fixed.csv'])
        self.n_control = settings.get('n_control', [5])
        self.niter = settings.get('niter', 5)
        self.thickness = settings.get('thickness', np.array([0, 0, 0.1]))
        self.stl_dir = settings.get('stl_dir', 'constant/triSurface/')
        self.stl_file_name = settings.get('stl_file_name', 'ribbon.stl')
        #import pdb; pdb.set_trace()
        self.setup()

    def setup(self, verbose=False):
        pts = [np.loadtxt(filename, delimiter=',') for filename in self.domain_files]
        fixed_points = [list(np.loadtxt(filename, delimiter=',').astype(int))\
                            for filename in self.fixed_points_files]
        cpolys = []
        for i in range(len(pts)):
            cpolys.append(support.ControlPolygon2D(pts[i], fixed_points[i], self.n_control[i]))
        ControlPolygonInterface.__init__(self, cpolys)
        problem = BasicPitzDailyRun(case_path=self.case_path)
        problem.prepare_case(self.source_case, verbose)
        self.problem = problem

    def info(self):
        raise NotImplementedError

    def get_configurable_settings(self):
        raise NotImplementedError

    def run(self, shape, verbose=False):
        curv_data = support.subdc_to_stl_mult(shape, self.niter,\
                    thickness=self.thickness,\
                    file_directory=self.case_path+self.stl_dir, \
                    file_name=self.stl_file_name,\
                    draw=False)
        p = self.problem.cost_function(verbose=verbose)
        if p == 0:
            raise Exception('Pressure difference is exactly zero. This is a bug related to OpenFoam.')
        return np.abs(p)

    def evaluate(self, decision_vector, verbose=False):
        if not self.constraint(decision_vector):
            raise ValueError('Constraint violated. Please supply a feasible decision vector.')
        shape = self.convert_decision_to_shape(decision_vector)
        return self.run(shape, verbose)

if __name__=='__main__':
    import numpy as np
    seed = 1435
    np.random.seed(seed)
    prob = PitzDaily({})
    lb, ub = prob.get_decision_boundary()
    x = np.random.random((1000, lb.shape[0])) * (ub - lb) + lb
    rand_x = []
    for i in range(x.shape[0]):
        if prob.constraint(x[i]):
            rand_x.append(x[i])
    res = prob.evaluate(rand_x[0])
    print(res)
