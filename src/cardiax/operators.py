"""

custom linear operators that extend lineax linear operators!!

"""

import jax.numpy as np
import jax.experimental.sparse as js

from lineax._tags import symmetric_tag, transpose_tags
import lineax as lx

class SparseMatrixLinearOperator(lx.MatrixLinearOperator):
    """Defines the matrix operator that will be used in the solver.
    The matrix object is given initially as a template for the object, and
    the dirichlet dofs are required because when solving Ax=b, we ignore
    the dofs that are fixed by Dirichlet boundary conditions.

    IMPORTANT: Once a discretization is chosen, the sparsity of the operator
    is fixed. Thus, once the operator is created, the data is the only thing that
    is updated in solver.

    Args:
        matrix (jax.experimental.sparse.BCOO): a batched COO matrix implemented in JAX,
            which is a sparse matrix with indices and data. For use in solver, this will be 
            initialized in a specific way.
        dirichlet_dofs (np.array): the indices of the Dirichlet boundary conditions.

    """
    dirichlet_dofs: np.array

    def __init__(self, matrix, dirichlet_dofs, tags=None):
        if tags is not None:
            super().__init__(matrix, tags)
        else:
            super().__init__(matrix)
        self.dirichlet_dofs = dirichlet_dofs
        # raise an exception if the dirichlet dofs are not integers
        # print(np.issubdtype(x.dtype, np.floating)) # Output: True
        if not np.issubdtype(dirichlet_dofs.dtype, np.integer):
            raise Exception("dirichlet_dofs must be a np.array of INTEGERS.")
        
        return

    def mv(self, dofs):
        """Specially defined matrix-vector multiplication for the solver.
        The sparsify function allows BCOO objects as input and removes the 0
        computations that are standard with JAX. The matvec operation also
        automatically 0s out appropriate values based on BCs to be used seamlessly
        with the solvers.

        Args:
            dofs (np.array): vector describing current state of the system

        Returns:
            res_vec (np.array): the result of the matrix-vector multiplication,
            if the data is the V matrix, then this will be the residual vector.
        """
        res_vec = js.sparsify(lambda m, v: m @ v)(self.matrix, dofs)
        res_vec = res_vec.at[self.dirichlet_dofs].set(dofs[self.dirichlet_dofs], unique_indices = True)
        return res_vec
    
    # issue with transpose: the mv method is overwritten when a linear opeartor is tranposed. idk if this is
    # an issue with lineax, or an issue with how we implemented sparsematrixlinearoperator...
    def transpose(self):
        if symmetric_tag in self.tags:
            return self
        # I don't think we use tags anywhere... but will still pass.
        # also, passing dirichlet dofs might cause issues if we are working with a non-square matrix?...
        return SparseMatrixLinearOperator(self.matrix.T, self.dirichlet_dofs, transpose_tags(self.tags))

