import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import os
import unittest

from cardiax import box_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
from cardiax import get_F, get_T, compute_traction
from cardiax.Lagrange.post_process import get_P

jax.config.update("jax_enable_x64", True)

# May want to change to use pressure formulation
# So there isn't volume locking issues

class HyperElasticity(Problem):
    def get_tensor_map(self):
        def psi(F):
            mu = 50.
            kappa = 1e3
            C1 = mu/2
            D1 = kappa/2
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = C1 * (I1 - 3. - 2 * np.log(J)) + D1 * (J - 1.)**2.
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

        gamma = 0.1
        def gamma_bc(point):
            return gamma

        def side_bc(point):
            return point[1] * gamma

        sides_bc = [side_bc, zero_bc, zero_bc]
        bc_left = [[left]*3, [0, 1, 2], sides_bc]
        bc_right = [[right]*3, [0, 1, 2], sides_bc]
        bc_front = [[front]*3, [0, 1, 2], [zero_bc]*3]
        bc_back = [[back]*3, [0, 1, 2], [gamma_bc, zero_bc, zero_bc]]
        bc_top = [[top]*3, [0, 1, 2], sides_bc]
        bc_bottom = [[bottom]*3, [0, 1, 2], sides_bc]
        dirichlet_bc_info = {"u": [bc_left, bc_right,
                            bc_front, bc_back,
                            bc_top, bc_bottom]}

        problem = HyperElasticity({"u": fe}, dirichlet_bc_info=dirichlet_bc_info)

        solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)), line_search_flag=True)
        sol, info = solver.solve(1e-8)

        Fs = get_F(problem.fes["u"], sol)
        F_analytic = np.array([[1., gamma, 0.],
                                [0., 1., 0.],
                                [0., 0., 1.]])
        onptest.assert_allclose(Fs.mean(axis=(0, 1)), F_analytic, atol=1e-8)
        
        stress_fn = problem.get_tensor_map()
        mu = 50.
        C1 = mu / 2
        T_analytic = np.array([[2 * C1 * gamma**2, 2 * C1 * gamma, 0.],
                            [2 * C1 * gamma, 0., 0.],
                            [0., 0., 0.]])
        Ts = get_T(sol, problem.internal_vars["u"], fe, stress_fn)
        onptest.assert_allclose(Ts.mean(axis=(0, 1)), T_analytic, atol=1e-8)

        P_analytic = T_analytic @ np.linalg.inv(F_analytic).T
        Ps = get_P(sol, problem.internal_vars["u"], fe, stress_fn)
        onptest.assert_allclose(Ps.mean(axis=(0, 1)), P_analytic, atol=1e-6)

        tract_left_analytic = P_analytic @ np.array([-1., 0., 0.])
        tract_left = compute_traction(sol, problem.internal_vars["u"], fe, stress_fn, left)
        onptest.assert_allclose(tract_left, tract_left_analytic, atol=1e-6)

        tract_right_analytic = P_analytic @ np.array([1., 0., 0.])
        tract_right = compute_traction(sol, problem.internal_vars["u"], fe, stress_fn, right)
        onptest.assert_allclose(tract_right, tract_right_analytic, atol=1e-6)

        tract_front_analytic = P_analytic @ np.array([0., -1., 0.])
        tract_front = compute_traction(sol, problem.internal_vars["u"], fe, stress_fn, front)
        onptest.assert_allclose(tract_front, tract_front_analytic, atol=1e-6)

        tract_back_analytic = P_analytic @ np.array([0., 1., 0.])
        tract_back = compute_traction(sol, problem.internal_vars["u"], fe, stress_fn, back)
        onptest.assert_allclose(tract_back, tract_back_analytic, atol=1e-6)

        tract_bottom_analytic = P_analytic @ np.array([0., 0., -1.])
        tract_bottom = compute_traction(sol, problem.internal_vars["u"], fe, stress_fn, bottom)
        onptest.assert_allclose(tract_bottom, tract_bottom_analytic, atol=1e-6)

        tract_top_analytic = P_analytic @ np.array([0., 0., 1.])
        tract_top = compute_traction(sol, problem.internal_vars["u"], fe, stress_fn, top)
        onptest.assert_allclose(tract_top, tract_top_analytic, atol=1e-6)

if __name__ == '__main__':
    unittest.main()