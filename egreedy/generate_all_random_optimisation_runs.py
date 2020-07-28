"""This script performs all "random" optimisation runs for each of the test
problems evaluated in the paper.
"""
from .util import generate_random_runs as grr


# double-check the user wants to overwrite any existing training data
value = input(
    'Enter "Y" to confirm you wish to generate all random'
    + "optimisation runs (any existing training data will be"
    + " overwritten): "
)
if value not in ["Y", "y"]:
    import sys

    sys.exit(0)

# parameters matching the paper - 51 total experiments per optimisation method
# and a budget of 250 function evaluations (including training data)
budget = 250
start_exp_no = 1
end_exp_no = 51
N_exps = end_exp_no - start_exp_no + 1

# synthetic test problems
problem_names = [
    "WangFreitas",
    "BraninForrester",
    "Branin",
    "GoldsteinPrice",
    "Cosines",
    "SixHumpCamel",
    "Hartmann6",
    "GSobol",
    "StyblinskiTang",
    "Rosenbrock",
    "logGoldsteinPrice",
    "logSixHumpCamel",
    "logHartmann6",
    "logGSobol",
    "logStyblinskiTang",
    "logRosenbrock",
]

for name in problem_names:
    grr.perform_LHS_runs(name, budget, start_exp_no, end_exp_no)

# robot pushing problems
for name in ["push4", "push8"]:
    grr.perform_LHS_runs(name, budget, start_exp_no, end_exp_no)

# PitzDaily CFD problem - This is uniformly sampled because it has a constraint
# function. Note that PitzDaily can only be ran on Linux.
print(
    "Now generating PitzDaily data, this will take approximately 195 hours"
    + " and can only be ran on Linux."
)
grr.perform_uniform_runs("PitzDaily", budget, start_exp_no, end_exp_no)
