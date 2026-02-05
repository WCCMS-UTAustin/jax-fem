import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import os
import unittest

from cardiax import box_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
from cardiax import get_grad

class HyperElasticity(Problem):
    def get_tensor_map(self):
        def psi(F):
            E = 1e2
            nu = 0.49
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(3)
            F = u_grad + I
            P = P_fn(F)
            return P
        return first_PK_stress

class Test(unittest.TestCase):
    """Test hyper-elasticity with cylinder mesh
    """

    def test_solve_problem(self):
        """Compare FEniCSx solution with JAX-FEM
        """
        problem_name = "nodal_stress"
        crt_dir = os.path.dirname(__file__)

        ele_type = 'hexahedron'
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        L, N = 1., 10
        mesh = box_mesh(N, N, N, L, L, L)
    
        def top(point):
            return np.isclose(point[1], L, atol=1e-5)

        def bottom(point):
            return np.isclose(point[1], 0., atol=1e-5)

        def left(point):
            return np.isclose(point[0], 0., atol=1e-5)

        def right(point):
            return np.isclose(point[0], L, atol=1e-5)
        
        def front(point):
            return np.isclose(point[2], 0., atol=1e-5)

        def back(point):
            return np.isclose(point[2], L, atol=1e-5)
                
        def zero_bc(point):
            return 0.

        def SS(point):
            return 0.4 * point[0]

        bc_left = [[right]*3, [0, 1, 2], [zero_bc, zero_bc, SS]]
        bc_right = [[left]*3, [0, 1, 2], [zero_bc]*3]
        bc_front = [[front]*3, [0, 1, 2], [zero_bc, zero_bc, SS]]
        bc_back = [[back]*3, [0, 1, 2], [zero_bc, zero_bc, SS]]
        bc_top = [[top]*3, [0, 1, 2], [zero_bc, zero_bc, SS]]
        bc_bottom = [[bottom]*3, [0, 1, 2], [zero_bc, zero_bc, SS]]

        dirichlet_bc_info = {"u": [bc_left, bc_right, bc_front, bc_back, bc_top, bc_bottom]}

        u_init = np.zeros_like(mesh.points)
        u_init = u_init.at[:, 2].set(0.35 * mesh.points[:, 0])
        
        fe = FiniteElement(mesh, vec=3, dim=3, ele_type="hexahedron", gauss_order=1)
        problem = HyperElasticity({"u": fe}, dirichlet_bc_info=dirichlet_bc_info)
        solver = Newton_Solver(problem, initial_guess=u_init, line_search_flag=True)
        sol, info = solver.solve()

        u_grad = get_grad(fe, sol)
        F = u_grad + np.eye(3)[None, None, :, :]
        F_true = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0.4, 0, 1]])
        diff = np.linalg.norm(F.mean(axis=(0, 1)) - F_true, ord="fro")
        self.assertTrue(diff < 1e-3)

if __name__ == '__main__':
    unittest.main()
