
from typing import Union
import jax
import jax.numpy as np
import functools as fctls
import time
from jaxtyping import ArrayLike

from cardiax._solver import Solver_Base
from cardiax import logger

class Newton_Solver(Solver_Base):
    """Full Newton Solver where we solve:
    D(u) \delta u = -r(u)
    At each step, D(u), the jacobian of the residual, r(u), is computed.
    This is done iteratively, linearizing the PDE and upating the solution until convergence

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

    # initial pass of reduced system testing: force user to
    # manually indicate if there are closed knots in a certain direction
    def solve(self, atol: float=1e-6, max_iter: int=30):
        """The solver imposes Dirichlet B.C. with "row elimination" method.

        Some memo:

        res(u) = D*r(u) + (I - D)u - u_b
        D = [[1 0 0 0]
            [0 1 0 0]
            [0 0 0 0]
            [0 0 0 1]]
        I = [[1 0 0 0]
            [0 1 0 0]
            [0 0 1 0]
            [0 0 0 1]
        A_fn = d(res)/d(u) = D*dr/du + (I - D)

        The function newton_update computes r(u) and dr/du
        """
        logger.debug(
            "Calling the row elimination solver for imposing Dirichlet B.C.")
        logger.debug("Start timing")
        start = time.time()

        self.res_norm_part = fctls.partial(self.res_norm_fn, internal_vars=self.problem.internal_vars,
                                    internal_vars_surfaces=self.problem.internal_vars_surfaces)

        dofs = np.zeros(self.problem.num_total_dofs_all_vars)

        if self.initial_guess is not None:
            dofs = jax.flatten_util.ravel_pytree(self.initial_guess)[0]

        res_vec, V = self.newton_update_helper(dofs, self.problem.internal_vars, self.problem.internal_vars_surfaces)
        res_val = np.linalg.norm(res_vec)
        res_val_init = res_val
        logger.debug(f"Before, res l_2 = {res_val}")
        counter = 0

        # track the total amount of time the linear solve and newton
        # updates take
        incremental_solve_total = 0
        # a newton update technically happens during the 0th iteration;
        # might want to also include that.
        newton_update_total = 0

        # save total time and use individual steps to better
        # break down the time
        while res_val > atol and counter < max_iter:
            dofs = self.linear_incremental_solver(res_vec, V, dofs)
            # newton update + timing
            res_vec, V = self.newton_update_helper(dofs, self.problem.internal_vars, self.problem.internal_vars_surfaces)
            res_val = np.linalg.norm(res_vec)                        
            logger.debug(f"res l_2 = {res_val}")
            counter += 1

            # # terminate if the residual is too large
            # # THIS IS DETERMINED HEURISTICALLY AS OF NOW
            # if res_val > 1e4:
            #     break
            # # should also exit if the jax linear solve doesn't coverge; that is a better test imo
            # if early_stop is not None:
            #     if res_val > res_val_init * early_stop:
            #         converged = False
            #         break

        if res_val <= atol and counter <= max_iter:
            converged = True
        else:
            converged = False

        end = time.time()
        solve_time = end - start
        logger.info(f"Solve took {solve_time} [s]")
        logger.info(f"Incremental Solves took {incremental_solve_total} [s]")
        logger.info(f"Newton Updates Took {newton_update_total} [s]")
        logger.debug(f"max of dofs = {np.max(dofs)}")
        logger.debug(f"min of dofs = {np.min(dofs)}")

        # return dofs and timing information
        return dofs, (converged, counter, solve_time, incremental_solve_total, newton_update_total, res_vec)
    