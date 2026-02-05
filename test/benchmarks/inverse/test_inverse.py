import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import os
import unittest

from cardiax import box_mesh
from cardiax import FiniteElement, Problem, Newton_Solver

jax.config.update("jax_enable_x64", True)

class HyperElasticity(Problem):
    def get_tensor_map(self):
        def psi(F, E):
            nu = 0.45
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy[0]
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, E):
            I = np.eye(3)
            F = u_grad + I
            P = P_fn(F, E)
            return P
        return first_PK_stress

    def get_surface_maps(self):

        def right_trac(u, u_grad, x):
            return np.array([-10., 0., 0.])

        return {"u": {"tract": right_trac}}

    def set_params(self, E):
        self.set_internal_vars({"u": {"E": self.fes["u"].convert_dof_to_quad(E)}})
        return

class Test(unittest.TestCase):
    def test_solve_problem(self):
        """ Test hyperelasticity analytic solution for shear
        """
        problem_name = "inverse"
        crt_dir = os.path.dirname(__file__)

        Lx, Ly, Lz = 1., 1., 1.
        ele_type = 'hexahedron'
        mesh = box_mesh(Nx=10, Ny=10, Nz=10, Lx=Lx, Ly=Ly, Lz=Lz)
        fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = ele_type, gauss_order = 1)

        def left(point):
            return np.isclose(point[0], 0., atol=1e-5)

        def right(point):
            return np.isclose(point[0], Lx, atol=1e-5)

        def zero_bc(point):
            return 0.

        bc_left = [[left]*3, [0, 1, 2], [zero_bc]*3]
        dirichlet_bc_info = {"u": [bc_left]}
        location_fns = {"u": {"tract": right}}

        # Set true values for stiffness
        problem = HyperElasticity({"u": fe}, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
        E_true = 80
        Es = E_true * np.ones((len(problem.mesh["u"].points), 1))
        problem.set_params(Es)

        # Determine the true displacement solution
        solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))
        sol_true, info = solver.solve(1e-8)

        # Wrap prediction to perform adjoint
        fwd_pred = solver.ad_wrapper()

        # Create error function for inverse problem
        def error(sol):
            return np.linalg.norm(sol - sol_true)

        def composed(E):
            sol = fwd_pred(E)
            return error(sol), sol

        val_grad = jax.value_and_grad(composed, has_aux=True)

        E_val = 50 # prediction of value
        E_guess = E_val * np.ones((len(problem.mesh["u"].points), 1))
        loss, tol = 1e3, 1e-2
        sol = np.zeros_like(problem.mesh["u"].points)
        while loss > tol: # Inverse solve loop
            solver.initial_guess = sol
            E_guess = E_val * np.ones((len(problem.mesh["u"].points), 1))
            vals = val_grad(E_guess)
            loss = vals[0][0]
            sol = vals[0][1]
            grad = vals[1].sum()
            print("################")
            print(f"Loss = {loss}")
            print(f"E: {E_val}")
            print("################")

            # Random guess of L for Newton
            E_val = E_val - loss/grad
            # Clip to speed up solve time without care for regularizing
            E_val = np.clip(E_val, 50, 100)

        print(f"E = {E_val}")
        diff = np.abs(E_val - E_true)
        print(f"E differs from analytic by {diff}")

        # Less than 1% error (error mostly comes from border elements)
        onptest.assert_almost_equal(diff, 0., decimal=4)

if __name__ == '__main__':
    unittest.main()

