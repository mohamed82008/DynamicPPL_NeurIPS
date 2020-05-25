# Setup Instructions

When first setting up this environment, please follow the following instructions:

1. Download and install Julia 1.4 from https://julialang.org/downloads/.
2. Download and setup PyStan using the instructions in https://pystan.readthedocs.io/en/latest/getting_started.html#.
3. Clone this repository in a directory on your system. Let's call the parent directory `xyz`.
4. `cd` (change directory) into the cloned repository using `cd xyz/DynamicPPL_NeurIPS` in your command line interface.
5. Call the Julia executable to start a new Julia session.
6. Run `using Pkg; Pkg.add("DrWatson"); Pkg.activate("."); Pkg.instantiate()`.
8. Setup `PyCall` by linking it against the Python version on your system used to install PyStan above. Instructions are provided in https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version.
9. Try the following code to check that all the above steps were successful:
```julia
using DynamicPPL, Turing, PyCall
pystan = pyimport("pystan")
```

After following the above steps, every time you want to run Julia code in this environment, only steps 5 and 6 above need to be followed.

# Benchmarks

Instructions for running the benchmarks are given [here](benchmarks/README.md).