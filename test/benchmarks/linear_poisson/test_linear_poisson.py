import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import os
import unittest

from cardiax import rectangle_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
from cardiax import get_F

class LinearPoisson(Problem):
    def get_tensor_map(self):
        return lambda x, f: x

    def get_mass_map(self):

        def forcing(u, x, u_grad, f):
            return -f

        return forcing

class Test(unittest.TestCase):
    """Test Poisson problem
    """
    def test_solve_problem(self):
        problem_name = "linear_poisson"

        N, L = 25, 1.
        mesh = rectangle_mesh(N, N, L, L, ele_type="quad8", degree=2)

        def left(point):
            return np.isclose(point[0], 0., atol=1e-5)

        def right(point):
            return np.isclose(point[0], L, atol=1e-5)
        
        def top(point):
            return np.isclose(point[1], 0., atol=1e-5)

        def bottom(point):
            return np.isclose(point[1], L, atol=1e-5)

        def zero_bc(point):
            return np.array([0.])

        bc_left = [[left]*3, [0, 1, 2], [zero_bc]*3]
        bc_right = [[right]*3, [0, 1, 2], [zero_bc]*3]
        bc_top = [[top]*3, [0, 1, 2], [zero_bc]*3]
        bc_bottom = [[bottom]*3, [0, 1, 2], [zero_bc]*3]

        dirichlet_bc_info = {"u": [bc_left, bc_right, bc_top, bc_bottom]}

        fe = FiniteElement(mesh, vec=1, dim=2, ele_type="quad8", gauss_order=2)
        problem = LinearPoisson({"u": fe}, dirichlet_bc_info=dirichlet_bc_info)
        solver = Newton_Solver(problem, np.zeros(problem.num_total_dofs_all_vars))
    
        def forcing_sin(x):
            return np.array([np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])])

        forcing = jax.vmap(forcing_sin)(mesh.points)
        forcing_quads = fe.convert_dof_to_quad(forcing)

        problem.set_internal_vars({"u": {"f": forcing_quads}})
        sol, info = solver.solve()

        # Test
        analytic_sol = forcing / (np.pi**2 * (2 ** 2 + 2 ** 2))

        # print(f"Solution absolute value differs by {np.max(np.absolute(jax_fem_sol - fenicsx_sol))} between FEniCSx and JAX-FEM")
        # onptest.assert_array_almost_equal(fenicsx_sol, jax_fem_sol, decimal=5)

        onptest.assert_array_almost_equal(analytic_sol.flatten(), sol, decimal=2)

if __name__ == '__main__':
    unittest.main()