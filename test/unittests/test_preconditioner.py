"""

    Tests preconditioners.

    Ensures that preconditioners are defined when they are
    requested to be, not defined when they are not, etc.

"""

import numpy as onp
import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)

# some testing classes to help us out
import unittest
from unittest.mock import patch, MagicMock
import numpy.testing as onptest

# solver class instances to test
from cardiax._solver import Solver_Base
from cardiax import Problem

# lineax imports to mimic downstream handling
# of preconditioner objects
from lineax._solver.misc import preconditioner_and_y0
import lineax as lx

class Test_Preconditioners(unittest.TestCase):
    """ Tests preconditioners
    
        Checks that default preconditioners behave as
        expected in CARDIAX. These tests will be expanded
        to provide users with a wider range of preconditioning
        options.

        We'll eventually want to test this on all solver class
        instances, but for now, we just want to test the behavior
        of the preconditioner used in `Solver_Base.jax_solve`.
    """

    @classmethod
    def setUpClass(cls):
    
        # 1. Define a test system to precondition
        #    these data structures will overwrite structures in solver
        A_np = onp.random.rand(20,20)  # size of problem does not matter
        cls.A_np = A_np
        cls.V, cls.I, cls.J = cls._get_VIJ_bcoo(A_np)             

        # 2. Define mock/test objects to create solver objects
        # NOTE: not sure how to handle this.
        init_guess = onp.zeros((len(A_np))) # hit initial guess with matrix rows!

        # get a mock problem object
        mock_problem_obj = cls._get_Mock_problem(init_guess, cls.I, cls.J)

        # 3. Define solver instances with and without preconditioners
        # problem: Problem
        # initial_guess: np.array 
        # precond: bool = field(default=True) # Should generalize to other preconditioners (maybe in different file eventually)
        # line_search_flag: bool = field(default=True)
        # use_petsc: bool = field(default=False)
        # petsc_options: list = field(default_factory=list) #Not sure what should be here
    
        cls.solver_with_pc = Solver_Base(mock_problem_obj, init_guess, precond='jacobi', line_search_flag=False) 
        cls.solver_no_pc = Solver_Base(mock_problem_obj, init_guess, precond='none', line_search_flag=False)

        # solver.A provides the structure of the LHS of our system so we can JIT-compile accordingly,
        # so we need to update the data here - it's filled with dummy variables.
        cls.solver_with_pc.A.matrix.data = cls.V
        cls.solver_no_pc.A.matrix.data = cls.V

    @classmethod
    def _get_VIJ_bcoo(self, A_np: onp.ndarray):
        """ returns sparse representation of a matrix

        Does not actually provide a sparse representation,
        just re-organizes the format to provide the data structures
        that would be used to create a BCOO representation of the
        dense matrix with no knowledge of non-zero entries.

        Parameters
        ----------
        A_np : onp.ndarray
            original dense matrix

        Returns
        -------
        tuple
            entry, row, and column of the matrix.
        """

        # for loop will suffice, there could be a better way to do this.
        V = onp.array([], dtype=onp.float64)
        I = onp.array([], dtype=onp.int32)
        J = onp.array([], dtype=onp.int32)

        # number of rows (and colums) in A
        n_row = A_np.shape[0]
        n_col = A_np.shape[1]
        
        # arrays to help populate I, J
        ones_array = onp.ones(n_row, dtype=onp.int32)
        cols_array = onp.arange(n_col, dtype=onp.int32)

        # for each row in A, populate V, I, J.
        for i in range(len(A_np)):
            V = onp.hstack((V, A_np[i]), dtype=onp.float64)
            I = onp.hstack((I, i * ones_array), dtype=onp.int32)
            J = onp.hstack((J, cols_array), dtype=onp.int32)

        return V, I, J

    # tool to help create a mock problem class
    # should this function livewithin Test_Preconditioners?
    @classmethod
    def _get_Mock_problem(self, init_guess, I, J):
        """ defines mock cardiax.Problem instance

        Parameters
        ----------
        init_guess : _type_
            _description_
        I : _type_
            _description_
        J : _type_
            _description_
        """
        
        # the MagicMock instance of the problem
        mock_problem = MagicMock(Problem)

        # quantities related to assembling the system of eq.
        mock_problem.num_total_dofs_all_vars = len(init_guess)
        mock_problem.I = np.array(I)
        mock_problem.J = J
        mock_problem.bc_inds = onp.array([], dtype=onp.int32) # no boundary conditions for now        

        # don't need any internal variables (right)?
        mock_problem.internal_vars = {'u':[]} 
        mock_problem.internal_vars_surfaces = {'u':[]}

        # try to type check certain things...
        # breakpoint()

        return mock_problem
    
    #############################################################
    ##                        tests !                          ##
    #############################################################
    # if this fails, PCs are broken!
    def test_pc_existence(self):
        """ tests default preconditioner existence

            NOTE: currently, self.get_preconditioner
            is the only function that provides preconditioners.
            If this is updated, this test will break! (that is ok).
        
        """
        # check that the diagonal / default preconditioner
        # has its expected type

        # note: this method emulates what happens in jax_solve,
        #       but doesn't call the method itself. I could maybe
        #       update this to do so, but for now, I don't think it
        #       makes sense to?

        # this code is repeated / might belong in setupclass!
        A_fn = self.solver_with_pc.A
        # will attempt altering this, need to chat with ben though.
        pc = self.solver_with_pc.get_preconditioner(A_fn.matrix)

        assert pc is not None

    # if this fails, PCs are broken!
    def test_pc_action(self):
        """ tests default preconditioner action

        """
        # check that the action of the preconditioner
        # (which is known as it's a diagonal preconditioner)
        # works as intended
        
        A_fn = self.solver_with_pc.A
        A_matrix = A_fn.as_matrix().todense()

        # get the true preconditioner, and its action
        pc = self.solver_with_pc.get_preconditioner(A_fn.matrix)

        # compare the test preconditioner to the action of the actual preconditioner
        cardiax_lineax_pc_action = np.zeros_like(A_matrix)
        # actual_pc_action = pc @ self.solver_with_pc.A
        for i in range (len(cardiax_lineax_pc_action)):
            cardiax_lineax_pc_action = cardiax_lineax_pc_action.at[:,i].set(pc.mv(A_matrix[:,i]))
        
        # compare the action of pc to the action of the diagonal
        # of A, which is used for the default jacobi preconditioner.
        A_diag_np = np.diag(self.A_np)
        pc_np = 1 / A_diag_np

        # create a matrix representation of the preconditioner
        pc_np_mat = onp.zeros(self.A_np.shape)
        diag_entries = onp.diag_indices(pc_np_mat.shape[0])
        pc_np_mat[diag_entries] = pc_np
        test_pc_action = pc_np_mat @ A_matrix

        # compare the action of the actual preconditioner with the action
        # of the manually constructed diagonal preconditioner.
        onptest.assert_allclose(test_pc_action, cardiax_lineax_pc_action)
    
    def test_no_pc_existence(self):
        """ tests 'None' preconditioner existence

        """
        # check that the preconditioner has the expected
        # type when no preconditioner is supplied
        A_fn = self.solver_no_pc.A
        pc = self.solver_no_pc.get_preconditioner(A_fn.matrix)
        
        # check that the preconditioner is the identity
        assert isinstance(pc, lx.IdentityLinearOperator) 
    

    # if this fails, PCs are broken!
    def test_no_pc_action(self):
        """ tests 'None' preconditioner action

        """
        # check that applying the preconditioner
        # doesn't change the matrix / functions
        # as the identity

        # multiply the preconditioner
        A_fn = self.solver_no_pc.A
        A_matrix = A_fn.as_matrix().todense()

        # get the true preconditioner, and its action
        pc = self.solver_no_pc.get_preconditioner(A_fn.matrix)
        
        # compare the test preconditioner to the action of the actual preconditioner
        cardiax_lineax_pc_action = np.zeros_like(A_matrix)
        # actual_pc_action = pc @ self.solver_with_pc.A
        for i in range (len(cardiax_lineax_pc_action)):
            cardiax_lineax_pc_action = cardiax_lineax_pc_action.at[:,i].set(pc.mv(A_matrix[:,i]))
            
        # the action of the pc on the stiffness matrix should be the identity.
        onptest.assert_allclose(A_matrix, cardiax_lineax_pc_action)

    @unittest.skip("Not Implemented Yet")
    def test_pc_failure(self):
        """ checks that we can catch an incorrectly
        supplied preconditioner.

        Using mock methods here can be helpful to manually override the
        code without actually modifying the methods themselves.
        """

        # overwrite the preconditioner method to supply the identity

        # assert that the preconditioner does not change the system,
        # which is not desired. If the diagonal of the stiffness matrix
        # is all ones, this behavior would actually be ok, so we're adding
        # a quick check to account for this.

    @unittest.skip("Not Implemented Yet")
    def test_no_pc_failure(self):
        """ checks that we can catch an incorrectly
        supplied preconditioner.

        Using mock methods here can be helpful to manually override the
        code without actually modifying the methods themselves.
        """

        # overwrite the preconditioner method to supply a non-identity
        # preconditioner object.

        # assert that the preconditioner 'fails' - that the application
        # of the preconditioner changes the system, which is not desired
        # in this case. 

    def test_solver_objs(self):
        """ tests test solver objects are defined correctly
        """

        # checks that solver instances' .precond value is
        # properly defined
        assert self.solver_no_pc.precond == 'none'
        assert self.solver_with_pc.precond == 'jacobi'

if __name__ == "__main__":
    unittest.main()