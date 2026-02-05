
from jax import device_get
import jax.numpy as np
from jaxtyping import ArrayLike
from typing import Union

from cardiax._solver import Solver_Base
import numpy as onp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
    
class Debug_Solver(Solver_Base):
    """Debug Solver 
    """

    def __post_init__(self):
        super().__post_init__()
        # self.newton_update_helper = jax.jit(self.newton_update_helper)
        return

    def newton_update_helper(self, dofs: ArrayLike, int_vars: Union[list, tuple], int_vars_surfs: Union[list, tuple]):
        """Function to create the residual vector and the jacobian of the residual

        Args:
            dofs (np.array): DOF array (base from the solver)
            int_vars (list): list with the internal variables used in the PDE
            int_vars_surfs (list): list of the internal variables on the surface used in the PDE

        Returns:
            np.array: residual vector
            np.array: the jacobian of the residual
        """
        res_vec, V = self.problem.newton_update_helper(dofs, int_vars, int_vars_surfs)
        res_vec = self.apply_bc_vec(res_vec, dofs)
        V = self.reduceV(V)
        return res_vec, V

    def visualize_mat(self, A):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(A.matrix.indices[:, 1], A.matrix.indices[:, 0], marker="s", s=4, color="black")
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("column")
        plt.ylabel("row")
        plt.title("Sparsity pattern (indices)")
        plt.show()

        return

    def get_cond_number(self, A):

        if hasattr(A, "matrix"):
            # gather data/indices from possibly-jax arrays onto CPU numpy
            data = A.matrix.data
            idx = A.matrix.indices
            rows = onp.asarray(idx[:, 0], dtype=onp.int64)
            cols = onp.asarray(idx[:, 1], dtype=onp.int64)

            m = int(rows.max()) + 1
            n = int(cols.max()) + 1

            M = coo_matrix((onp.asarray(data), (rows, cols)), shape=(m, n)).tocsr()
        else:
            M = A.tocsr()

        # Try a sparse SVD for large matrices, fallback to dense SVD for reliability
        # largest singular value
        _, s_max, _ = svds(M, k=1, which="LM", tol=1e-8)
        # smallest singular value
        _, s_min, _ = svds(M, k=1, which="SM", tol=1e-8)
        max_sv = float(s_max[0])
        min_sv = float(s_min[0])
        cond = float(abs(max_sv) / abs(min_sv)) if abs(min_sv) > 0 else onp.inf

        return cond
    
    def create_linear_system(self, u):
        res_vec, V = self.newton_update_helper(u, self.problem.internal_vars, self.problem.internal_vars_surfaces)
        A_fn = self.A
        A_fn.matrix.data = V
        A_fn.matrix.indices = np.vstack([self.I, self.J]).T
        return A_fn, res_vec

    def create_precond_system(self, u):
        A_fn, res_vec = self.create_linear_system(u)
        pc = self.get_jacobi_precond(A_fn.matrix)

        # Not sure how to handle the FunctionalLinearOperator pc
        # First makes them dense then back to sparse...
        precond = pc.as_matrix() @ A_fn.as_matrix()

        return coo_matrix(precond)
