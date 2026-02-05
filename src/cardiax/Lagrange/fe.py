
import numpy as onp
import jax.numpy as np
import time
from cardiax._basis import get_face_shape_vals_and_grads, get_shape_vals_and_grads
from cardiax._fe import Base_FE
from cardiax import logger
from meshio import Mesh
from jaxtyping import ArrayLike
import jax

class FiniteElement(Base_FE):
    """This class defines the tools used to describe the discretized function space
    over the mesh that is used as input.

    Args:
        mesh (meshio.Mesh): The mesh object containing the points and cells
        vec (int): Number of components of the solution vector
        dim (int): Dimension of the problem, e.g. 3 for 3D
        ele_type (str): Type of the element, e.g. 'hexahedron'
        gauss_order (int): Order of the Gauss quadrature rule to use for integration
        
    Attributes:
        mesh (meshio.Mesh): The mesh object containing the points and cells
        points (np.ndarray): coordinates of the mesh points
        nodes (np.ndarray): same as points, used for boundary conditions
        cells (np.ndarray): connectivity of the mesh cells
        num_cells (int): number of cells in the mesh
        num_total_nodes (int): total number of nodes in the mesh
        num_total_dofs (int): total number of degrees of freedom (nodes * vec)
        shape_vals (np.ndarray): shape function values at quadrature points
        shape_grads_ref (np.ndarray): gradients of the shape functions in reference coordinates
        quad_weights (np.ndarray): weights for the Gauss quadrature rule
        face_shape_vals (np.ndarray): shape function values on the faces of the elements
        face_shape_grads_ref (np.ndarray): gradients of the shape functions on the faces in reference coordinates
        face_quad_weights (np.ndarray): weights for the Gauss quadrature rule on the faces
        face_normals (np.ndarray): normals of the faces of the elements
        face_inds (np.ndarray): indices of the faces in the mesh
        num_quads (int): number of quadrature points    
    
    """

    def __init__(self, mesh: Mesh, vec: int, dim: int, ele_type: str, gauss_order: int):
        """_summary_

        Args:
            mesh (Mesh): The mesh object containing the points and cells
            vec (int): Number of components of the solution vector
            dim (int): Dimension of the problem, e.g. 3 for 3D
            ele_type (str): Type of the element, e.g. 'hexahedron'
            gauss_order (int): Order of the Gauss quadrature rule to use for integration
        """

        self.mesh = mesh
        super().__init__(vec, dim, ele_type, gauss_order)

        self.points = self.mesh.points[:, :self.dim]
        # not sure if this is accurate nomenclature-wise, but need to
        # access different objects to apply boundary conditions for iga and FE
        self.nodes = self.mesh.points[:, :self.dim]
        self.cells = self.mesh.cells_dict[self.pv_ele_type].astype(np.int32)
        self.num_cells = len(self.cells)
        self.num_total_nodes = len(self.mesh.points)
        self.num_total_dofs = self.num_total_nodes * self.vec

        start = time.time()
        logger.debug("Computing shape function values, gradients, etc.")

        self.shape_vals, self.shape_grads_ref, self.quad_weights = get_shape_vals_and_grads(self.ele_type, self.dim * self.gauss_order)
        self.face_shape_vals, self.face_shape_grads_ref, self.face_quad_weights, self.face_normals, self.face_inds \
        = get_face_shape_vals_and_grads(self.ele_type, self.dim * self.gauss_order)
        self.num_quads = self.shape_vals.shape[0]
        self.num_nodes = self.shape_vals.shape[1]
        self.num_faces = self.face_shape_vals.shape[0]
        self.shape_grads, self.JxW = self.get_shape_grads()

        # (num_cells, num_quads, num_nodes, 1, dim)
        self.v_grads_JxW = self.shape_grads[:, :, :, None, :] * self.JxW[:, :, None, None, None]
        self.num_face_quads = self.face_quad_weights.shape[1]
        self.cell_shape_vals = np.repeat(self.shape_vals[None,:,:], self.num_cells, axis = 0)

        end = time.time()
        compute_time = end - start

        logger.debug(f"Done pre-computations, took {compute_time} [s]")
        logger.info(f"Solving a problem with {len(self.cells)} cells, {self.num_total_nodes}x{self.vec} = {self.num_total_dofs} dofs.")
        return

    def local_to_cell_dofs(self, local_dofs: jax.Array) -> jax.Array:
        """ Converts local dofs to cell dofs

        Args:
            local_dofs (np.ndarray): Local degrees of freedom

        Returns:
            np.ndarray: Cell degrees of freedom
        """

        sol = local_dofs.reshape((self.num_total_nodes, self.vec))
        return sol[self.cells]

    def cells_dof_to_sol(self,
                         cell_dofs: jax.Array, 
                         cell_index: int) -> jax.Array:
        """ Converts cell degrees of freedom to solution vector.
        This is identity for Lagrange elements.
        
        Args:
            cell_dofs (np.ndarray): Cell degrees of freedom
            cell_index (int): Index of the cell

        Returns:
            np.ndarray: Solution vector
        """
        return cell_dofs

    def get_cell_basis_supports(self) -> ArrayLike:
        """ Returns the basis function supports for each cell.
        This is identity for Lagrange elements.

        Returns:
            np.ndarray: Basis function supports for each cell
        """
        return self.cells

    def get_cell_dof_supports(self) -> ArrayLike:
        """_summary_

        Args:
            var (_type_): _description_

        Returns:
            _type_: _description_
        """

        cell_dof_map = onp.zeros((self.num_cells, self.num_nodes*self.vec))
        for index, cell in enumerate(self.cells):
            cell_dof_map[index] = onp.hstack(onp.array([onp.arange(entry*self.vec, entry*self.vec + self.vec) for entry in cell]))
        return cell_dof_map.astype(onp.int32)

    def convert_dof_to_quad(self, var: jax.Array) -> jax.Array:
        """ Converts DoFs to quadrature points. Needed if internal variables
        are defined on the nodes to construct the residual.

        Args:
            var (np.ndarray): Degrees of freedom values

        Returns:
            np.ndarray: Quadrature point values
        """

        var_cells = np.take(var, self.cells, axis=0)
        var_quads = np.sum(self.shape_vals[None, :, :, None] * var_cells[:, None, :, :], axis=2)
        return var_quads
