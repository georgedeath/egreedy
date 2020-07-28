"""This script randomly samples 10^5 evaluations of each problem instance of
the push8 test problem, evaluates them, and runs a local optimiser (L-BFGS-B)
on the best 100 of them. This is in an attempt to estimate the global optimum
of the function to compare to the Bayesian optimisation runs for each
acquisition function.

The results of each set of evaluates are compared to the best seen (lowest)
function evaluation across all techniques, and the best of these are taken as
the estimated global optimum.

The final results of this are saved to 'results/push8_best_solutions.npz',
containing an numpy.ndarray of shape (51, ) named 'push8_best' with values that
the corresponding best seen values for that problem number, eg push8_best[i]
corresponds to the problem i-start_exp_no; where start_exp_no is typically 1.
"""
import glob
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from . import test_problems

value = input(
    'Enter "Y" to confirm you wish to potentially overwrite'
    + "the push8 evaluations found: "
)
if value not in ["Y", "y"]:
    import sys

    sys.exit(0)

# number of random decision vectors to generate
N_TOTAL = int(1e5)

# top N_BEST of these to evaluate locally optimise with L-BFGS-B
N_BEST = 100

# start and end experiment numbers
start_exp_no = 1
end_exp_no = 51

results = np.zeros(end_exp_no - start_exp_no + 1)

data_name = r"training_data/push8_{:d}.npz"

for problem_no in range(1, 52):
    # load the problem inputs that define the targets
    with np.load(data_name.format(problem_no), allow_pickle=True) as data:
        inputs = data["arr_2"].item()

    # instantiate the function
    f_class = test_problems.push8
    f = f_class(**inputs)

    # create and evaluate decision vectors
    # NOTE: here we cannot use LHS because generating 1e5 samples is too large
    # (in terms of memory) for our computing capability
    # from pyDOE2 import lhs
    # X = lhs(f.dim, samples=N, criterion='maximin')
    X = np.random.uniform(size=(N_TOTAL, f.dim)) * (f.ub - f.lb) + f.lb
    y = f(X).ravel()

    # evaluate the top N_BEST with L-BFGS-B
    sorted_inds = np.argsort(y)[:N_BEST]

    # L-BFGS-B bounds: tuples of lower/upper bound pairs for each dim
    bounds = list(zip(f.lb, f.ub))

    # set up the results saving
    results[problem_no - start_exp_no] = np.min(y)

    # run L-BFGS-B on each of the N_BEST solutions to try to improve them
    for i, idx in enumerate(sorted_inds):
        xb, yb = X[idx, :], y[idx]
        xbnew, ybnew, _ = fmin_l_bfgs_b(f, x0=xb, bounds=bounds, approx_grad=True)
        yb = np.minimum(yb, ybnew)
        if yb < results[problem_no - start_exp_no]:
            results[problem_no - start_exp_no] = yb

    # finally, compare the best fitness value seen from all optimisation runs
    # with the best found from the uniform sampling + local optimisation
    results_mask = f"results/push8_{problem_no:d}_*.npz"
    result_files = glob.glob(results_mask)

    # store the best value of each run
    for result_file in result_files:
        with np.load(result_file) as data:
            min_Y = np.min(np.squeeze(data["Ytr"]))
            if min_Y < results[problem_no - start_exp_no]:
                results[problem_no - start_exp_no] = min_Y

    print("Finished evaluating push8 problem {:d}".format(problem_no))

    np.savez("training_data/push8_best_solutions.npz", results=results)
