
import jax
import jax.numpy as np
import time
from jaxtyping import ArrayLike

from cardiax._solver import Solver_Base
from cardiax import logger


class Grad_Solver(Solver_Base):
    """
    Performs gradient descent which optimizes naively by 
    updating by the gradient times a step size alpha.
    Not recommended for standard FE, use Newton instead.
    """
    
    def __post_init__(self):
        super().__post_init__()
        # self.newton_update_helper = jax.jit(self.newton_update_helper)
        self.get_res_grad = jax.jit(jax.grad(self.loss_fn, argnums=0))
        self.alpha = 1e-4
        self.dirichlet_dofs, self.dirichlet_vals = self.problem.get_boundary_data()
        return

    def loss_fn(self, dofs: ArrayLike, internal_vars: ArrayLike, internal_vars_surfaces: ArrayLike) -> float:
        res_vec = self.problem.compute_residual_helper(dofs, internal_vars, internal_vars_surfaces)
        res_vec = res_vec.at[self.dirichlet_dofs].set(dofs[self.dirichlet_dofs])
        res_vec = res_vec.at[self.dirichlet_dofs].add(-self.dirichlet_vals, unique_indices=True)
        return np.linalg.norm(res_vec)

    # initial pass of reduced system testing: force user to
    # manually indicate if there are closed knots in a certain direction
    def solve(self, atol: float=1e-6, max_iter: int=30, early_stop: bool=None):

        logger.debug(
            "Calling the row elimination solver for imposing Dirichlet B.C.")
        logger.debug("Start timing")
        start = time.time()

        dofs = np.zeros(self.problem.num_total_dofs_all_vars)

        if self.initial_guess is not None:
            dofs = jax.flatten_util.ravel_pytree(self.initial_guess)[0]

        res_vec = self.problem.compute_residual_helper(dofs, self.problem.internal_vars, self.problem.internal_vars_surfaces)
        res_val = np.linalg.norm(res_vec)
        res_val_init = res_val
        logger.debug(f"Before, res l_2 = {res_val}")
        counter = 0

        # save total time and use individual steps to better
        # break down the time
        while res_val > atol and counter < max_iter:
            grads = self.get_res_grad(dofs, self.problem.internal_vars, self.problem.internal_vars_surfaces)
            dofs = dofs - self.alpha * grads
            dofs = self.assign_bc(dofs)
            res_vec = self.problem.compute_residual_helper(dofs, self.problem.internal_vars, self.problem.internal_vars_surfaces)
            res_val = np.linalg.norm(res_vec)                        
            logger.debug(f"res l_2 = {res_val}")
            counter += 1

            # terminate if the residual is too large
            # THIS IS DETERMINED HEURISTICALLY AS OF NOW
            if res_val > 1e4:
                break
            # should also exit if the jax linear solve doesn't coverge; that is a better test imo
            if early_stop is not None:
                if res_val > res_val_init * early_stop:
                    converged = False
                    break

        if res_val <= atol and counter <= max_iter:
            converged = True
        else:
            converged = False

        end = time.time()
        solve_time = end - start
        logger.info(f"Solve took {solve_time} [s]")

        # return dofs and timing information
        return dofs, (converged, counter, solve_time, res_vec)