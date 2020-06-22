"""
Evaluates all methods on the a set of test functions functions.

Examples:
    Evaluate all methods in the paper on the synthetic functions:
    > python run_all_experiments.py synthetic

    Evaluate all methods in the paper on the robot pushing functions:
    > python run_all_experiments.py robot

    Evaluate all methods in the paper on the PitzDaily test function:
    (Note that this can only be performed if OpenFOAM is set up correctly)
    > python run_all_experiments.py pitzdaily

optional arguments:
  -h, --help            show this help message and exit
  -f {synthetic,robot,pitzdaily}
                        Set of test problems to evaluate.
"""
import argparse
from egreedy.optimizer import perform_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''
Evaluate all methods on a set of functions.
--------------------------------------------
Examples:
    Evaluate all methods in the paper on the synthetic functions:
    > python run_all_experiments.py synthetic

    Evaluate all methods in the paper on the robot pushing functions:
    > python run_all_experiments.py robot

    Evaluate all methods in the paper on the PitzDaily test function:
    (Note that this can only be performed if OpenFOAM has been set up correctly)
    > python run_all_experiments.py pitzdaily
''')

    parser.add_argument(dest='problem_types',
                        type=str,
                        choices=['synthetic', 'robot', 'pitzdaily'],
                        help='Set of test problems to evaluate.')

    # parse the args so they appear as a.argname, eg: a.budget
    a = parser.parse_args()

    if a.problem_types == 'synthetic':
        problem_names = ['WangFreitas', 'BraninForrester', 'Branin', 'Cosines',
                         'logGoldsteinPrice', 'logSixHumpCamel', 'modHartman6',
                         'logGSobol', 'logStyblinskiTang', 'logRosenbrock']

    elif a.problem_types == 'robot':
        problem_names = ['push4', 'push8']

    elif a.problem_types == 'pitzdaily':
        problem_names = ['PitzDaily']

    method_names = ['Explore', 'EI', 'PI', 'UCB', 'PFRandom', 'Exploit',
                    'eRandom_eps0.01', 'eRandom_eps0.05', 'eRandom_eps0.1',
                    'eRandom_eps0.2', 'eRandom_eps0.3', 'eRandom_eps0.4',
                    'eRandom_eps0.5',
                    'eFront_eps0.01', 'eFront_eps0.05', 'eFront_eps0.1',
                    'eFront_eps0.2', 'eFront_eps0.3', 'eFront_eps0.4',
                    'eFront_eps0.5']

    budget = 250

    # evaluate each method 51 times on each problem
    for problem in problem_names:
        for method in method_names:
            acquisition_args = {}

            # parse the epsilon value for the e-greedy methods
            if '_eps' in method:
                method, eps = method.split('_eps')
                acquisition_args = {'epsilon': float(eps)}

            # run the experiment (1 to 51)
            for run_no in range(1, 52):
                perform_experiment(problem,
                                   run_no,
                                   method,
                                   acquisition_args=acquisition_args,
                                   budget=budget,
                                   continue_runs=True,
                                   verbose=True,
                                   save=True)
