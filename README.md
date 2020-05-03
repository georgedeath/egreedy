
## Greed is Good: Exploration and Exploitation Trade-offs in Bayesian Optimisation
#### George De Ath <sup>1</sup>, Richard M. Everson <sup>1</sup>, Alma A. M. Rahat, S <sup>2</sup>, Jonathan E. Fieldsend <sup>1</sup>
<sup>1</sup> University of Exeter, United Kingdom, <sup>2</sup> Swansea University, United Kingdom

This repository contains the Python3 code for the ε-greedy strategies presented in:
> George De Ath, Richard M. Everson, Alma A. M. Rahat, and Jonathan E. Fieldsend. 2020. Greed is Good: Exploration and Exploitation Trade-offs in Bayesian Optimisation, to appear in ACM Transactions on Evolutionary Learning and Optimization (TELO).
> **Preprint:** https://arxiv.org/abs/1911.12809

The repository also contains all training data used for the initialisation of each of the 51 optimisation runs carried to evaluate each method, the optimisation results of each of the runs on each of the methods evaluated and the code to generate new training data and also to perform the optimisation runs themselves. Two jupyter notebooks are also included that reproduce all figures shown in the paper and supplementary material.

The remainder of this document details:
- The steps needed to install the package and related python modules on your system: [docker](#installation-docker) / [manual](#installation-manual)
- The format of the [training data](#training-data) and [saved runs](#optimisation-results).
- How to [repeat the experiments](#reproduction-of-experiments).
- How to [reproduce the figures in the paper](#reproduction-of-figures-and-tables-in-the-paper).
- How to [add your own acquisition functions and test problems](#incorporation-of-additional-test-problems-and-acquisition-functions).

### Citation
If you use any part of this code in your work, please cite our [Arxiv paper](https://arxiv.org/abs/1911.12809):
```bibtex
@misc{death:egreedy,
    title = {Greed is Good: Exploration and Exploitation Trade-offs in Bayesian Optimisation},
    author = {George {De Ath} and Richard M. Everson and Alma A. M. Rahat and Jonathan E. Fieldsend},
    year = {2019},
    eprint = {arXiv:1911.12809}
}
```

### Installation (docker)
The easiest method to automatically set up the environment needed for the optimisation library to run and to repeat the experiments carried out in this work is to use [docker](http://www.docker.com). Install instructions for docker for many popular operating systems are can be found [here](https://docs.docker.com/install/). Once docker has been installed, the docker container can be download and ran as follows:
```bash
> # download the docker container
> docker pull georgedeath/egreedy
> # run the container
> docker run -it georgedeath/egreedy /bin/bash
Welcome to the OpenFOAM v5 Docker Image
..
```
Once the above commands have been ran you will be in the command prompt of the container, run the following commands to test the functionality of the code (CTRL+C to prematurely halt the run):
```bash
> # change to the code directory - this contains the git repo
> cd /egreedy
> # run an example optimisation run
> python -m egreedy.optimizer
Loaded training data from: training_data/Branin_1.npz
Loaded test problem: Branin
Using acquisition function: eFront
        with optional arguments: {'epsilon': 0.25}
Training a GP model with 4 data points.
Optimising the acquisition function.
..
```


### Installation (manual)
Manual installation is straight-forward for the optimisation library apart from the configuration of the PitzDaily test problem due to the installation and compilation of [OpenFOAM®](http://www.openfoam.com). Note that if you do not wish to use the PitzDaily test problem then the library will work fine without the optional instructions included at the end of this section. The following instructions will assume that [Anaconda3](https://docs.anaconda.com/anaconda/install/) has been installed and that you are running the following commands from the command prompt/console:

```bash
> conda install -y scipy numpy matplotlib statsmodels swig jupyter
> conda install -y pygmo --channel conda-forge
> pip install nlopt pyDOE2 pygame box2d-py GPy numpy-stl
```
Note that, on windows, to install `swig` and `pygame` it may be necessary to also install [Visual C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

Once the above python modules have been installed, clone this repository to a location of your choosing (in the following we assume you are installing to `/egreedy/`) and test that it works (CTRL+C to cancel optimisation run):
```bash
> git clone https://github.com/georgedeath/egreedy/ /egreedy
> cd /egreedy
> python -m egreedy.optimizer
Loaded training data from: training_data/Branin_1.npz
..
```
PitzDaily (CFD) instructions (**optional, Linux only**) - other test problems will work without this:
```
$ pip install pyfoam 
```
Now follow the linked instructions to [install OpenFOAM5](https://openfoamwiki.net/index.php/Installation/Linux/OpenFOAM-5.x/Ubuntu) (this will take 30min - 3hours to install). Note that this has only been tested with the Ubuntu 12.04 and 18.04 instructions. Once this has been successfully installed, the command `of5x` has to be ran before the PitzDaily test problem can be evaluated.

Finally, compile the pressure calculation function and check that the test problem works correctly:
```bash
> of5x
> cd /egreedy/egreedy/test_problems/Exeter_CFD_Problems/data/PitzDaily/solvers/
> wmake calcPressureDifference
> # test the PitzDaily solver
> cd /egreedy
> python -m egreedy.test_problems.pitzdaily
PitzDaily successfully instantiated..
Generated valid solution, evaluating..
Fitness value: [0.24748876]
```
Please ignore errors like `Getting LinuxMem: [Errno 2] No such file or directory: '/proc/621/status` as these are from OpenFOAM and do not impact the optimisation process.

### Training data
The initial training locations for each of the 51 sets of [Latin hypercube](https://www.jstor.org/stable/1268522) samples are located in the `training_data` directory in this repository with the filename structure `ProblemName_number`, e.g. the first set of training locations for the Branin problem is stored in `Branin_1.npz`. Each of these files is a compressed numpy file created with [numpy.savez](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html). It has two [numpy.ndarrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html) containing the 2*D initial locations and their corresponding fitness values. To load and inspect these values use the following instructions:
```python
> cd /egreedy
> python
>>> import numpy as np
>>> with np.load('training_data/Branin_1.npz') as data:
	Xtr = data['arr_0']
	Ytr = data['arr_1']
>>> Xtr.shape, Ytr.shape
((4, 2), (4, 1))
```
The robot pushing test problems (push4 and push8) have a third array `'arr_2'`  that contains their instance-specific parameters:
```python
> cd /egreedy
> python
>>> import numpy as np
>>> with np.load('training_data/push4_1.npz', allow_pickle=True) as data:
	Xtr = data['arr_0']
	Ytr = data['arr_1']
	instance_params = data['arr_2']
>>> instance_params
array({'t1_x': -4.268447250704135, 't1_y': -0.6937799887556437}, dtype=object)
```
these are automatically passed to the problem function when it is instantiated to create a specific problem instance.

### Optimisation results
The results of all optimisation runs can be found in the `results` directory. The filenames have the following structure: `ProblemName_Run_TotalBudget_Method.npz`, with the ε-greedy methods having the format: `ProblemName_Run_TotalBudget_Method_eps0.XX.npz` where `XX` corresponds to the value of ε used. Similar to the training data, these are also [numpy.ndarrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)  and contain two items, `Xtr` and `Ytr`, corresponding to the evaluated locations in the optimisation run and their function evaluations. Note that the evaluations and their function values will also include the initial 2*D training locations at the beginning of the arrays and that the methods ε-RS and ε-PF have results files named *eRandom* and *eFront* respectively.

The following example loads the first optimisation run on the Branin test problem with the ε-PF method using ε = 0.1:
```python
> cd /egreedy 
> python
>>> import numpy as np
>>> # load the 
>>> with np.load('results/Branin_1_250_eFront_eps0.1.npz', allow_pickle=True) as data:
	Xtr = data['Xtr']
	Ytr = data['Ytr']
>>> Xtr.shape, Ytr.shape
((250, 2), (250, 1))
```

### Reproduction of experiments
The python file `run_experiment.py` provides a convenient way to reproduce an individual experimental evaluation carried out the paper. It has the following syntax:
```bash
> python run_experiment.py -h
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
```
Similarly, `run_all_experiments.py` provides an easy interface run **all** experiments for a specific set of test problems, either the synthetic, robot pushing or pipe shape optimisation, in the following manner:
```bash
> python run_all_experiments.py -h
usage: run_all_experiments.py [-h] {synthetic,robot,pitzdaily}

Evaluate all methods on a set of functions.
--------------------------------------------
Examples:
    Evaluate all methods in the paper on the synthetic functions:
    > python run_all_experiments.py -f synthetic

    Evaluate all methods in the paper on the robot pushing functions:
    > python run_all_experiments.py -f robot

    Evaluate all methods in the paper on the PitzDaily test function:
    (Note that this can only be performed if OpenFOAM has been set up correctly)
    > python run_all_experiments.py -f pitzdaily

positional arguments:
  {synthetic,robot,pitzdaily}
                        Set of test problems to evaluate.
```
Note that each test problem is evaluated approximately 250000 times for the 20 methods in the script. The synthetic functions and robot pushing problem have trivial computational cost but the Gaussian Processes need to be trained and corresponding acquisition function optimised for each function evaluation. The PitzDaily test problem, however, will take around 1 minute to evaluate, meaning that the total time spent evaluating the computational fluid dynamics solver is  approximately 160 days. Given that the optimisation runs are independent and embarrassingly parallel, we recommend the use of ``run_experiment.py`` in a batch setting across multiple cores/machines.

### Reproduction of figures and tables in the paper
The [jupyter](https://jupyter.org) notebook [Non_results_figure_generation.ipynb](notebooks/New_fitness_functions_and_acquisition_functions.ipynb) contains the code to generate the following figures:
- Figure 1: Showing an example Gaussian process model and its corresponding Pareto front and set.
- Figure 2: Contours of acquisition function values for EI, UCB and PI.
- Figure 3: Contours of weighted EI for three values of ω.
- Figure 1 (Supplementary material): Landscape of the WangFreitas test problem.

The jupyter notebook [Process_results_and_generate_figures_for_paper.ipynb](notebooks/Process_results_and_generate_figures_for_paper.ipynb) contains the code to load and process the optimisation results (stored in the `results` directory) as well as the code to produce all results figures and tables used in the paper and supplementary material.

### Incorporation of additional test problems and acquisition functions
The jupyter notebook [New_fitness_functions_and_acquisition_functions.ipynb](notebooks/New_fitness_functions_and_acquisition_functions.ipynb) contains examples and instructions of how to include your own test problems (fitness functions) and acquisition functions.
