import jax.numpy as np
import jax
from cardiax import Problem

class PDE(Problem):
    """This is a neo-Hookean material model.

    Args:
        Problem (_type_): _description_
    """
    def get_tensor_map(self):

        def psi(F):
            mu = self.E / (2. * (1. + self.nu))
            kappa = self.E / (3. * (1. - 2. * self.nu))
            J = np.linalg.det(F)
            Jinv = 1/(np.cbrt(J**2))
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I_d = np.eye(u_grad.shape[0])
            F = u_grad + I_d
            P = P_fn(F)
            return P

        return first_PK_stress

    def set_params(self, params):
        self.E = params.get('E', 10.)
        self.nu = params.get('nu', 0.3)
        return