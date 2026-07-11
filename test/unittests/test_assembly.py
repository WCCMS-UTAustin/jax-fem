"""

Tests the assembly of RHS and LHS of various systems.



"""

# unit testing imports
import pytest
import jax.numpy as np
import jax
import numpy.testing as onptest
import numpy as onp

# cardiax-related imports
from cardiax import (
    FiniteElement,
    Problem,
    Newton_Solver
)
from cardiax import box_mesh
from cardiax.PDEs.Mechanics.Hyperelasticity.NeoHookean import PDE

######################################################
#################### testing suite! ##################
######################################################

# test parameterization, by element type
parameterize_element_types = pytest.mark.parametrize(
    "element_type_list",
    [
        # [ele_type, gauss_order]
        ["hexahedron", 1], 
        ["hexahedron27", 3]
    ]
)
# would like to parameterize by class type too, where
# different Problem classes are used and tested.
#
param_PDEs = pytest.mark.parametrize(
    "pde",
    [
        "<problem class name 1>",
        "<problem class name 2>"
    ]
)

###################################################
## problem class that prescribes a body force ##
###################################################
class PDE_w_body_force(PDE):
    """ Adds body force 

    Adds a body force to emulate gravity

    """

    # need to update the actual value!
    def get_mass_map(self):
        def mass_map(u, u_grad, x):
            val = np.array([0.0, -1.0, 0.0]) 
            return val
        return mass_map

# class-based fixtures, which aren't quite what I need
# https://stackoverflow.com/questions/28288385/pytest-how-to-pass-an-argument-to-setup-class

# (not sure if this is what I really want, the above answer makes more sense)
# https://stackoverflow.com/questions/38729007/parametrize-class-tests-with-pytest

# We don't need a solver class at all in this test.
# This speeds up the test by 2x (for small example).
class Assembly_Helper:

    def __init__(self, element_type_list, body_force=False):

        # process parameterization
        element_type = element_type_list[0]
        gauss_order = element_type_list[1]

        # 0. get mesh - single element.
        Nx = 1
        Ny = 1
        Nz = 1
        Lx = 1
        Ly = 1
        Lz = 1

        # (will need to be custom method for TETs)
        mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz,
                    ele_type=element_type)

        # 1. get FE object
        self.fe = FiniteElement(mesh, 3, 3, element_type, gauss_order)

        # 2. get Problem instance
        # dirichlet_bcs = get_boundary_conditions(element_type)
            
        # probably only need to apply a displacement to all nodes.
        def u_no_disp(point):
            return 0.0

        # if we're applying this bc to all nodes, we don't
        # need to distinguish between the IGA and FE cases.
        def x0_nodes(point):
            return np.allclose(point[0], 0.0)

        # define dirichlet boundary condition data structure
        bcs_list = [[x0_nodes], [0], [u_no_disp]]
        dirichlet_bcs = {'u': [bcs_list]}

        # 3. create problem class instances

        # augment the PDE if we want to apply a body force
        # (Ben's input file code handles this well, don't do this
        #  in practice!)
        if body_force:
            self.problem = PDE_w_body_force({'u': self.fe}, dirichlet_bc_info=dirichlet_bcs)
        else:
            self.problem = PDE({'u': self.fe}, dirichlet_bc_info=dirichlet_bcs)
        self.problem.set_params({'E': 10.0, 'nu': 0.3}) # use default values

        # # 3. get solver instance
        # self.u0 = np.zeros_like(self.problem.fes['u'].nodes).flatten()
        # # iga solver
        # newton_solver_params = {'problem': self.problem,
        #                         'initial_guess': self.u0,
        #                         'line_search_flag': True,
        #                         'precond':'jacobi'
        #                     }
        # self.solver = Newton_Solver(**newton_solver_params)

# this fails to re-use the same problem and solver
# class. However, it might make much more sense to use
# subTests in the context of test parameterization here.
#
# hopefully there's a better way to do this.
@parameterize_element_types
def test_RHS(element_type_list):

    # generate FE, Problem, and Solver class instances
    assembly_helper = Assembly_Helper(element_type_list)

    # (find a way to parameterize this)
    # assembly with a different PDE
    assembly_helper_body = Assembly_Helper(element_type_list, body_force=True)


    # perturb a few dofs
    dofs = np.zeros_like(assembly_helper.fe.nodes.flatten())
    key = jax.random.key(1)
    noise = jax.random.normal(key, dofs.shape[0]) * 1e-4
    dofs = dofs + noise

    # might need to initialize these
    int_vars = assembly_helper.problem.internal_vars
    int_vars_surfs = assembly_helper.problem.internal_vars_surfaces

    # call various solver methods and check that their
    # outputs match expected analytical solutions.
    res_vec, V = assembly_helper.problem.newton_update_helper(dofs,
                                                          int_vars,
                                                          int_vars_surfs)

    # check that the residual is non-zero, which is trivial with
    # noise.
    onptest.assert_(onp.linalg.norm(res_vec) > dofs.shape[0] * 1e-16)

    # also perturb the problem with the body force
    # might need to initialize these
    int_vars_b = assembly_helper_body.problem.internal_vars
    int_vars_surfs_b = assembly_helper_body.problem.internal_vars_surfaces

    # call various solver methods and check that their
    # outputs match expected analytical solutions.
    res_vec_b, V_b = assembly_helper_body.problem.newton_update_helper(dofs,
                                                          int_vars_b,
                                                          int_vars_surfs_b)

    # check that res_vec is different (it obviously will be)
    onptest.assert_(onp.linalg.norm(res_vec - res_vec_b) > dofs.shape[0] * 1e-16)

    # compare res_vec to an analytical RHS.