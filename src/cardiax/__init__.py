import importlib.metadata
from jax import config
import os

__version__ = importlib.metadata.version(__package__)

from ._logger_setup import setup_logger
# LOGGING
logger = setup_logger(__name__)

def set_jax_enable_x64(enable: bool = True):
    """
    Enable or disable JAX 64-bit mode.
    Args:
        enable (bool): If True, enable 64-bit mode. If False, disable it.
    """
    config.update('jax_enable_x64', enable)
    # automatically choose the highest matmul precision
    config.update('jax_default_matmul_precision', 'highest')
    logger.info(f"Set jax_enable_x64 to {enable}")

def logger_off():
    """
    Turn off logging for the cardiax package.
    """
    logger.setLevel("CRITICAL")
    return

# Set default
set_jax_enable_x64(True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

############## Lagrange Capabilities #################
from .Lagrange.fe import FiniteElement as FiniteElement

############## Important Classes #################
from ._problem import Problem as Problem
from .solvers.newton import Newton_Solver as Newton_Solver
from .solvers.gradient_descent import Grad_Solver as Grad_Solver

############### Mesh Generation ###################
from .Lagrange.generate_mesh import (
    rectangle_mesh as rectangle_mesh,
    box_mesh as box_mesh,
    sphere_mesh as sphere_mesh,
    hollow_sphere_mesh as hollow_sphere_mesh,
    ellipsoid_mesh as ellipsoid_mesh,
    hollow_ellipsoid_mesh as hollow_ellipsoid_mesh,
    cylinder_mesh as cylinder_mesh,
    hollow_cylinder_mesh as hollow_cylinder_mesh,
    prolate_spheroid_mesh as prolate_spheroid_mesh,
)

############### Postprocessing ###################
from .utils import save_sol as save_sol
from .Lagrange.post_process import (
    get_grad as get_grad,
    get_F as get_F,
    get_C as get_C,
    get_E as get_E,
    get_P as get_P,
    get_T as get_T,
    surface_integral as surface_integral,
    compute_traction as compute_traction,
    get_surface_normals_nodes_current as get_surface_normals_nodes_current,
)