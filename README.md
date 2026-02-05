
# CARDIAX

CARDIAX is a GPU-accelerated differentiable finite element package based on [JAX](https://github.com/google/jax), forked from [JAX-FEM](https://github.com/deepmodeling/jax-fem). CARDIAX focuses on solving cardiac mechanics problems in a single, unified framework from Laplace-Dirichlet-Rule-Based method for fiber generation to biventricular inverse models. This package is actively managed by the [Willerson Center for Cardiovascular Modeling and Simulation (WCCMS)](https://oden.utexas.edu/research/centers-and-groups/willerson-center-for-cardiovascular-modeling-and-simulation/) and is constantly adapting to accommodate the suite of problems we are intereseted in solving. **We are only focused on GPU development**.

<!--
Change the website once updated:
https://oden.utexas.edu/research/centers-and-groups/willerson-center-for-cardiovascular-modeling-and-simulation/
to
https://wccms.oden.utexas.edu/
-->

## Installation

Before installing `cardiax` be sure to install `jax` at [JAX Install](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-locally-harder). Verify that the GPU is seen by running the following to see if CUDA devices are found

```python
import jax
print(jax.devices())
```

Once the `jax` installation is working, the easiest option is to build all the dependencies through a conda environment using `environment.yaml` which also installs JAX with CUDA. Another option is to build a virtual environment with venv and pip for the dependencies. We have a `SC_requirements.txt` for installing on the supercomputer systems (TACC) where only venv is allowed. This can be modified to run on your own cluster if there are strict versioning requirements. These files set up the dependecies, but to install `cardiax`, you must go inside the directory `../CARDIAX` and run

```
pip install -e .
```

**Warning: JAX just upgrade to 0.9, so we haven't created the latest environment yet.**

## Examples

In the documentation, there are examples that walk through how to use the code. These are under tutorials, but the files are markdown format to explain functionality. The corresponding `*.py` files live in the `CARDIAX/tutorials` directory. More examples are also given in the demos but without the lengthy markdown explanations.

## Limitations

### Coding
While JAX supports CPU, CARDIAX is not being tested on CPU environments. We created this codebase to fully leverage GPUs, but the functionality should remain 
consistent. Also, multi-GPU functionality is also not available. The problems we are currently solving can fit on the memory of a single GPU, so we will not develop this parallelization until needed.

### Finite Element

The finite element limitations are quite vast compared to some other codes. Not to say these can't be addressed, but they would take some work. This repo is consistently being developed, but here are some problems we don't support:

- Mixed FE fields with off-diagonal block entries
- Fluid-structure interaction
- Manipulation of test function

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE](https://www.gnu.org/licenses/) for details.

## Citations

If you're using this project, you can cite this work [here](CITATION.cff).

We'll add a list of others papers built upon this framework below:



