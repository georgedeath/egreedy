"""
Command-line parser for running experiments.

usage: run_experiment.py [-h] -p PROBLEM_NAME -b BUDGET -r RUN_NO -a
                         ACQUISITION_NAME
                         [-aa ACQUISITION_ARGS [ACQUISITION_ARGS ...]]

egreedy optimisation experimental evaluation
--------------------------------------------
Example:
    Running the ePF method on the Branin test function with the training data
    "1" for a budget (including 2*D training points) of 250 and with a value
    of epsilon = 0.1 :
    > python run_experiment.py -p Branin -b 250 -r 1 -a eFront -aa epsilon:0.1

    Running EI on push4 method (note the lack of -aa argument):
    > python run_experiment.py -p push4 -b 250 -r 1 -a EI

optional arguments:
  -h, --help            show this help message and exit
  -p PROBLEM_NAME       Test problem name. e.g. Branin, logGSobol
  -b BUDGET             Budget. Default: 250 (including training points). Note
                        that the corresponding npz file containing the initial
                        training locations must exist in the "training_data"
                        directory.
  -r RUN_NO             Run number
  -a ACQUISITION_NAME   Acquisition function name. e.g: Explore, EI, PI UCB,
                        PFRandom, eRandom (e-RS), eFront (e-PF) or Exploit
  -aa ACQUISITION_ARGS [ACQUISITION_ARGS ...]
                        Acquisition function parameters, must be in pairs of
                        parameter:values, e.g. for the e-greedy methods:
                        epsilon:0.1 [Note: optional]
"""
import argparse
from egreedy.optimizer import perform_experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
egreedy optimisation experimental evaluation
--------------------------------------------
Example:
    Running the ePF method on the Branin test function with the training data
    "1" for a budget (including 2*D training points) of 250 and with a value
    of epsilon = 0.1 :
    > python run_experiment.py -p Branin -b 250 -r 1 -a eFront -aa epsilon:0.1

    Running EI on push4 method (note the lack of -aa argument):
    > python run_experiment.py -p push4 -b 250 -r 1 -a EI
""",
    )

    parser.add_argument(
        "-p",
        dest="problem_name",
        type=str,
        help="Test problem name. e.g. Branin, logGSobol",
        required=True,
    )

    parser.add_argument(
        "-b",
        dest="budget",
        type=int,
        help="Budget. Default: 250 (including training points)."
        + " Note that the corresponding npz file"
        + " containing the initial training locations"
        + ' must exist in the "training_data" directory.',
        required=True,
    )

    parser.add_argument("-r", dest="run_no", type=int, help="Run number", required=True)

    parser.add_argument(
        "-a",
        dest="acquisition_name",
        type=str,
        help="Acquisition function name. e.g: Explore, EI, PI"
        + " UCB, PFRandom, eRandom (e-RS), eFront (e-PF)"
        + " or Exploit",
        required=True,
    )

    parser.add_argument(
        "-aa",
        dest="acquisition_args",
        nargs="+",
        help="Acquisition function parameters, must be in "
        + "pairs of parameter:values,"
        + " e.g. for the e-greedy methods:"
        + " epsilon:0.1 [Note: optional]",
        required=False,
    )

    # parse the args so they appear as a.argname, eg: a.budget
    a = parser.parse_args()

    # convert the acq func args into a dict
    acquisition_args = {}
    if a.acquisition_args is not None:
        for kv in a.acquisition_args:
            k, v = kv.split(":")

            try:
                acquisition_args[k] = float(v)
            except ValueError:
                acquisition_args[k] = v

    # run the experiment
    perform_experiment(
        a.problem_name,
        a.run_no,
        a.acquisition_name,
        acquisition_args=acquisition_args,
        budget=a.budget,
        continue_runs=True,
        verbose=True,
        save=True,
    )
