import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import os
import unittest

from cardiax import box_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
from cardiax import get_F

jax.config.update("jax_enable_x64", True)

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
    def test_solve_problem(self):
        """ Test hyperelasticity analytic solution for shear
        """
        problem_name = "hyperelastic_shear"
        crt_dir = os.path.dirname(__file__)

        Lx, Ly, Lz = 1., 1., 1.
        ele_type = 'hexahedron'
        mesh = box_mesh(Nx=15, Ny=15, Nz=15, Lx=Lx, Ly=Ly, Lz=Lz)
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 1)

        def left(point):
            return np.isclose(point[0], 0., atol=1e-5)

        def right(point):
            return np.isclose(point[0], Lx, atol=1e-5)

        def front(point):
            return np.isclose(point[1], 0., atol=1e-5)

        def back(point):
            return np.isclose(point[1], Ly, atol=1e-5)

        def bottom(point):
            return np.isclose(point[2], 0., atol=1e-5)

        def top(point):
            return np.isclose(point[2], Lz, atol=1e-5)

        def zero_bc(point):
            return 0.

        lamb = 0.1
        def y0_bc(point):
            return -lamb/2

        def y1_bc(point):
            return lamb/2

        def z0_bc(point):
            return -.5*(1/(1+lamb) - 1)

        def z1_bc(point):
            return .5*(1/(1+lamb) - 1)

        bc_left = [[left], [0], [zero_bc]]
        bc_right = [[right], [0], [zero_bc]]
        bc_front = [[front], [1], [y0_bc]]
        bc_back = [[back], [1], [y1_bc]]
        bc_bottom = [[bottom], [2], [z0_bc]]
        bc_top = [[top], [2], [z1_bc]]
        dirichlet_bc_info = {"u": [bc_left, bc_right,
                            bc_front, bc_back, 
                            bc_top, bc_bottom]}

        problem = HyperElasticity({"u": fe}, dirichlet_bc_info=dirichlet_bc_info)

        solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))
        sol, info = solver.solve(1e-10)

        F = get_F(problem.fes["u"], sol)
        F_analytic = np.array([[1, 0, 0],
                            [0., 1.1, 0],
                            [0, 0, 1/(1+.1)]])

        diff = np.linalg.norm(F - F_analytic, axis=(2, 3), ord="fro").mean()

        print(f"CARDIAX F differs from analytic by {diff}")

        # Less than 1% error (error mostly comes from border elements)
        onptest.assert_almost_equal(diff, 0., decimal=8)

if __name__ == '__main__':
    unittest.main()
