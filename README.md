# Setup Instructions

When first setting up this environment, please follow the following instructions. Steps 2 and 7 can be ignored if you don't want to run Stan models.

1. Download and install Julia 1.4 from https://julialang.org/downloads/.
2. Clone this repository in a directory on your system. Let's call the parent directory `xyz`.
3. `cd` (change directory) into the cloned repository using `cd xyz/DynamicPPL_NeurIPS` in your command line interface.
4. Run the Julia executable to start a new Julia session.
5. Run `using Pkg; Pkg.add("DrWatson"); Pkg.activate("."); Pkg.instantiate()`.

## Installation of PyStan
There are two possibilities to install PyStan.

#### Using PIP
1. Download and setup PyStan using the instructions in https://pystan.readthedocs.io/en/latest/getting_started.html#.
2. Setup `PyCall` by linking it against the Python version on your system used to install PyStan above. Instructions are provided in https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version.

#### Using Conda
In the running Julia session, execute `using Conda; Conda.add("pystan=2.19.1.1", channel="conda-forge");` and re-open the Julia session.

## Final Test 
You should now be ready to run the experiments.
Try the following code to check that all the above steps were successful:
```julia
using DynamicPPL, Turing, PyCall
pystan = pyimport("pystan")
pystan.__version__ # this should give you 2.19.1.1
```

The above steps are only needed the first time when setting things up. Every time after that when you want to run some Julia code in this environment, follow the steps below:
1. `cd` (change directory) into the cloned repository using `cd xyz/DynamicPPL_NeurIPS` in your command line interface.
2. Call the Julia executable to start a new Julia session,
3. Run `using DrWatson; @quickactivate "DynamicPPL_NeurIPS"`

# Benchmarks

Instructions for running the benchmarks are given [here](benchmarks/README.md).
