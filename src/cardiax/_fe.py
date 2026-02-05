
from typing import Callable
import numpy as onp
import jax
import jax.numpy as np
import pyvista as pv
from dataclasses import dataclass
from jaxtyping import ArrayLike

from abc import ABC, abstractmethod

@dataclass
class Base_FE(ABC):
    """
    Base class to be used for FE objects. Current implementation is made to
    work with traditional Lagrange FE, but hopefully, can be extended to other types.
    ie future work on potentially Bezier extraction, IGA, or others...
    The ABC framework insure consistency when initializing the
    different classes because they will be shared 
    with Problem.

    Attributes
    ----------
    vec : int
        Number of vector components (e.g., 1 for scalar, 3 for vector fields).
    dim : int
        Spatial dimension (e.g., 2 for 2D problems, 3 for 3D problems).
    ele_type : str
        Element type (e.g., 'hexahedron', 'quad').
    gauss_order : int
        Order of the Gauss quadrature rule used for numerical integration.    
    """

    vec: int
    dim: int
    ele_type: str
    gauss_order: int

    def __post_init__(self):
        self.pv_ele_type = self.meshio_to_pyvista_ele(self.ele_type)
        return


    def meshio_to_pyvista_ele(self, ele_type: str):
        """Convert meshio element type to pyvista element type.
        """
        if ele_type == 'triangle':
            pv_ele_type = pv.CellType.TRIANGLE
        elif ele_type == 'quadrilateral' or ele_type == 'quad':
            pv_ele_type = pv.CellType.QUAD
        elif ele_type == "quad8":
            pv_ele_type = pv.CellType.QUADRATIC_QUAD
        elif ele_type == 'tetrahedron' or ele_type == 'tetra':
            pv_ele_type = pv.CellType.TETRA
        elif ele_type == 'hexahedron' or ele_type == 'hex':
            pv_ele_type = pv.CellType.HEXAHEDRON
        elif ele_type == 'hexahedron27' or ele_type == 'hex27':
            pv_ele_type = pv.CellType.TRIQUADRATIC_HEXAHEDRON
        else:
            raise NotImplementedError(f"Element type {ele_type} not implemented in FE class.")
        return pv_ele_type

    @abstractmethod
    def local_to_cell_dofs(self, local_dofs: ArrayLike):
        """Returns the local indices that contribute to cell dofs

        Returns:
            _type_: _description_
        """
        return

    @abstractmethod
    def cells_dof_to_sol(self, cell_dofs: ArrayLike, cell_index: int):
        """Add documentation

        Returns:
            _type_: _description_
        """

        return

    @abstractmethod
    def get_cell_basis_supports(self):
        """Add documentation

        Returns:
            _type_: _description_
        """
        return

    @abstractmethod
    def get_cell_dof_supports(self):
        """Add documentation

        Returns:
            _type_: _description_
        """
        return
    
    @abstractmethod
    def convert_dof_to_quad(self, var: ArrayLike):
        """Add documentation

        Returns:
            _type_: _description_
        """
        return
    
    def get_sol_shape(self):
        """Add documentation

        Returns:
            _type_: _description_
        """
        return (self.num_total_nodes, self.vec)

    def get_shape_grads(self):
        """Compute shape function gradient value
        The gradient is w.r.t physical coordinates.
        See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
        Page 147, Eq. (3.9.3)

        Returns
        -------
        shape_grads_physical : onp.ndarray
            (num_cells, num_quads, num_nodes, dim)
        JxW : onp.ndarray
            (num_cells, num_quads)
        """
        assert self.shape_grads_ref.shape == (self.num_quads, self.num_nodes, self.dim)
        physical_coos = onp.take(self.points[:, :self.dim], self.cells, axis=0)  # (num_cells, num_nodes, dim)
        # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
        jacobian_dx_deta = onp.sum(physical_coos[:, None, :, :, None] *
                                   self.shape_grads_ref[None, :, :, None, :], axis=2, keepdims=True)
        jacobian_det = onp.linalg.det(jacobian_dx_deta)[:, :, 0]  # (num_cells, num_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)
        # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim)
        # (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, dim)
        shape_grads_physical = (self.shape_grads_ref[None, :, :, None, :]
                                @ jacobian_deta_dx)[:, :, :, 0, :]
        JxW = jacobian_det * self.quad_weights[None, :]
        return np.array(shape_grads_physical), np.array(JxW)

    def get_face_shape_grads(self, boundary_inds):
        """Face shape function gradients and JxW (for surface integral)
        Nanson's formula is used to map physical surface ingetral to reference domain
        Reference: https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

        Parameters
        ----------
        boundary_inds : List[onp.ndarray]
            (num_selected_faces, 2)

        Returns
        -------
        face_shape_grads_physical : onp.ndarray
            (num_selected_faces, num_face_quads, num_nodes, dim)
        nanson_scale : onp.ndarray
            (num_selected_faces, num_face_quads)
        """
        physical_coos = onp.take(self.points[:, :self.dim], self.cells, axis=0)  # (num_cells, num_nodes, dim)
        selected_coos = physical_coos[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_f_shape_grads_ref = self.face_shape_grads_ref[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes, dim)
        selected_f_normals = self.face_normals[boundary_inds[:, 1]]  # (num_selected_faces, dim)

        # (num_selected_faces, 1, num_nodes, dim, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        # (num_selected_faces, num_face_quads, num_nodes, dim, dim) -> (num_selected_faces, num_face_quads, dim, dim)
        jacobian_dx_deta = onp.sum(selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :], axis=2)
        jacobian_det = onp.linalg.det(jacobian_dx_deta)  # (num_selected_faces, num_face_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)  # (num_selected_faces, num_face_quads, dim, dim)

        # (1, num_face_quads, num_nodes, 1, dim) @ (num_selected_faces, num_face_quads, 1, dim, dim)
        # (num_selected_faces, num_face_quads, num_nodes, 1, dim) -> (num_selected_faces, num_face_quads, num_nodes, dim)
        face_shape_grads_physical = (selected_f_shape_grads_ref[:, :, :, None, :] @ jacobian_deta_dx[:, :, None, :, :])[:, :, :, 0, :]

        # (num_selected_faces, 1, 1, dim) @ (num_selected_faces, num_face_quads, dim, dim)
        # (num_selected_faces, num_face_quads, 1, dim) -> (num_selected_faces, num_face_quads)
        nanson_scale = onp.linalg.norm((selected_f_normals[:, None, None, :] @ jacobian_deta_dx)[:, :, 0, :], axis=-1)
        # extra thing to return to try to scale a load by face areas
        # area = nanson_scale # need to check if this is actually the area
        selected_weights = self.face_quad_weights[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads)
        nanson_scale = nanson_scale * jacobian_det * selected_weights
        # return face_shape_grads_physical, nanson_scale, area
        return np.array(face_shape_grads_physical), np.array(nanson_scale)

    def get_physical_quad_points(self):
        """Compute physical quadrature points

        Returns
        -------
        physical_quad_points : np.ndarray
            (num_cells, num_quads, dim)
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)
        # (1, num_quads, num_nodes, 1) * (num_cells, 1, num_nodes, dim) -> (num_cells, num_quads, dim)
        physical_quad_points = onp.sum(self.shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2)
        return np.array(physical_quad_points)

    def get_physical_surface_quad_points(self, boundary_inds):
        """Compute physical quadrature points on the surface

        Parameters
        ----------
        boundary_inds : List[onp.ndarray]
            ndarray shape: (num_selected_faces, 2)

        Returns
        -------
        physical_surface_quad_points : ndarray
            (num_selected_faces, num_face_quads, dim)
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)
        selected_coos = physical_coos[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, num_face_quads, num_nodes, 1) * (num_selected_faces, 1, num_nodes, dim) -> (num_selected_faces, num_face_quads, dim)
        physical_surface_quad_points = onp.sum(selected_face_shape_vals[:, :, :, None] * selected_coos[:, None, :, :], axis=2)
        return np.array(physical_surface_quad_points)

    def get_surface_normals(self, location_fn: Callable):
        """ Obtain the outward normals on a given boundary.

        Args:
            location_fn (Callable): A function that flags if a point is on the boundary.

        Returns:
            normals (np.ndarray): Outward normals on the boundary.
        """

        bndry = self.get_boundary_conditions_inds(location_fn)

        physical_coos = onp.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        selected_coos = physical_coos[bndry[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_f_shape_grads_ref = self.face_shape_grads_ref[bndry[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes, dim)

        jacobian_dx_deta = onp.sum(selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :], axis=2)
        # jacobian_det = onp.linalg.det(jacobian_dx_deta)  # (num_selected_faces, num_face_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)  # (num_selected_faces, num_face_quads, dim, dim)
        selected_f_normals = self.face_normals[bndry[:, 1]]  # (num_selected_faces, dim)
        normals = (selected_f_normals[:, None, None, :] @ jacobian_deta_dx)[:, :, 0, :]
        return normals/np.linalg.norm(normals, axis=-1)[:, :, None]
    
    def get_dirichlet_data(self, mark_fcs, components, value_fcs):
        """Add documentation

        Args:
            degree (_type_): _description_

        Returns:
            _type_: _description_
        """

        node_inds_list, vec_inds_list, vals = self.Dirichlet_boundary_conditions((mark_fcs, components, value_fcs))
        
        # going to write unit test that checks that the local_inds generated
        # are accessing the correct DOF.
        local_inds = onp.array(node_inds_list)*self.vec + onp.array(vec_inds_list)
        
        return local_inds, vals
    
    def Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Indices and values for Dirichlet B.C.

        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]

        Returns
        -------
        node_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to num_total_nodes - 1
        vec_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to to vec - 1
        vals_List : List[ndarray]
            Dirichlet values to be assigned
        """
        node_inds_list = []
        vec_inds_list = []
        vals_list = []
        if dirichlet_bc_info is not None:
            location_fns, vecs, value_fns = dirichlet_bc_info
            assert len(location_fns) == len(value_fns) and len(value_fns) == len(vecs)
            for i, location_fn in enumerate(location_fns):
                node_inds = onp.argwhere(jax.vmap(location_fn)(self.nodes)).reshape(-1)
                vec_inds = onp.ones_like(node_inds, dtype=onp.int32) * vecs[i]
                values = jax.vmap(value_fns[i])(self.nodes[node_inds].reshape(-1, self.dim)).reshape(-1)
                node_inds_list.append(node_inds)
                vec_inds_list.append(vec_inds)
                vals_list.append(values)
        return node_inds_list, vec_inds_list, vals_list
    
    def get_boundary_conditions_inds(self, location_fn):
        """Given location functions, compute which faces satisfy the condition.

        Parameters
        ----------
        location_fns : List[Callable]
            Callable: a location function that inputs a point (ndarray) and returns if the point satisfies the location condition
                      e.g., lambda x: np.isclose(x[0], 0.)
                      If this location function takes 2 arguments, then the first is point and the second is index.
                      e.g., lambda x, ind: np.isclose(x[0], 0.) & np.isin(ind, np.array([1, 3, 10]))

        Returns
        -------
        boundary_inds_list : List[onp.ndarray]
            (num_selected_faces, 2)
            boundary_inds_list[k][i, 0] returns the global cell index of the ith selected face of boundary subset k
            boundary_inds_list[k][i, 1] returns the local face index of the ith selected face of boundary subset k
        """
        cell_points = onp.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        cell_face_points = onp.take(cell_points, self.face_inds, axis=1)  # (num_cells, num_faces, num_face_vertices, dim)
        # cell_face_inds = onp.take(self.cells, self.face_inds, axis=1) # (num_cells, num_faces, num_face_vertices)
        if location_fn is not None:
            vmap_location_fn = jax.vmap(location_fn)
            def on_boundary(cell_points):
                boundary_flag = vmap_location_fn(cell_points)
                return onp.all(boundary_flag)

            vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
            boundary_flags = vvmap_on_boundary(cell_face_points)
            boundary_inds = onp.argwhere(boundary_flags)  # (num_selected_faces, 2)

        return boundary_inds

    def update_Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Reset Dirichlet boundary conditions.
        Useful when a time-dependent problem is solved, and at each iteration the boundary condition needs to be updated.

        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]
        """
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(dirichlet_bc_info)
        return
