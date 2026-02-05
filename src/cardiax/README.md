# CARDIAX

CARDIAX is a generic finite element package; however, we are focusing the development of this package on valve mechanics and electromechanics of the heart. These two problems are the motivation for continued work and give the direction for future development. None of this would be possible without the original JAX-FEM repository, started by Tianjue Xue found here: https://github.com/deepmodeling/jax-fem. We have forked from this repo and tailored the code towards the interest of the WCCMS group. The major changes that have been made to the codebase are as follows:

1. Refactoring the code to become more independent
   1. FE object contains information about the mesh and the specific variable field
   2. Problem object contains the information for the boundary conditions, the PDE to solve, and how to combine this information with FE to set up the global system
   3. Solver is now class based, making it more akin to Fenicsx, allowing the solver to be manipulated while performing iterative solves.
2. Through the refactor, we have abstracted classes to handle various FE implementations that may 
be attempted in the future.