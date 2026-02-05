
import jax
import jax.numpy as np
import numpy as onp
from jax.typing import ArrayLike
from jax import Array
from cardiax import FiniteElement
from typing import Callable

def calc_grads(cell_dofs, shape_grads):
    """Computes the gradient from cell_dofs and shape_grads

    Args:
        cell_dofs (ArrayLike): The cell degrees of freedom.
        shape_grads (ArrayLike): The shape function gradients.

    Returns:
        Array: The computed gradients.
    """
    return np.einsum('ji,kjl->kil', cell_dofs, shape_grads)

def get_grad(fe: FiniteElement, sol: ArrayLike) -> Array:
    """Computes the gradient of the solution.

    Args:
        fe (FiniteElement): The finite element object.
        sol (ArrayLike): The solution vector.

    Returns:
        Array: The gradient of the solution.
    """

    cell_dofs = fe.local_to_cell_dofs(sol)
    grads = jax.vmap(calc_grads)(cell_dofs, fe.shape_grads)
    return grads

def get_F(fe: FiniteElement, sol: ArrayLike) -> Array:
    """Retrieves the deformation gradient from the solution at
       quadrature points in the mesh

    Args:
        fe (FiniteElement class): Contains mesh information for FE solve
        sol (np.array): the flattened solution vector of displacements

    Returns:
        F (np.array): deformation gradient of shape (num_cells, num_quds, dim, dim)
    """
    u_grads = get_grad(fe, sol)
    F = u_grads + np.full_like(u_grads, np.eye(fe.dim)[None, None, :, :])
    return F

def get_C(fe: FiniteElement, sol: ArrayLike) -> Array:
    """Returns the Right Cauchy Green tensor

    Args:
        fe (FiniteElement class): Contains mesh information for FE solve
        sol (np.array): the flattened solution vector of displacements

    Returns:
        C (np.array): Right Cauchy Green tensor of shape (num_cells, num_quds, dim, dim)
    """
    F = get_F(fe, sol)
    C = np.transpose(F, axes=(0, 1, 3, 2)) @ F
    return C

def get_E(fe: FiniteElement, sol: ArrayLike) -> Array:
    """Returns the Green-St. Venant strain tensor

    Args:
        fe (FiniteElement class): Contains mesh information for FE solve
        sol (np.array): the flattened solution vector of displacements

    Returns:
        E (np.array): Green-St. Venant strain tensor of shape (num_cells, num_quds, dim, dim)
    """
    C = get_C(fe, sol)
    E = 1/2 * (C - np.full_like(C, np.eye(fe.dim)[None, None, :, :]))
    return E

def get_P(sol: ArrayLike, internal_vars: dict, 
          fe: FiniteElement, P_func: Callable):

    int_vars = [internal_vars[var] for var in internal_vars]
    P = jax.vmap(jax.vmap(P_func))(get_grad(fe, sol), *int_vars)
    return P

def get_T(sol: ArrayLike, internal_vars: dict, 
          fe: FiniteElement, P_func: Callable) -> Array:
    """Compute the Cauchy Stress Tensor over the cells

    Args:
        fe (FiniteElement): Contains mesh information for FE solve
        P_func (Callable): The first_Pk functions defined in the problem class
        sol (ArrayLike): The solution vector
        internal_vars (list): Internal variables needed for the stress computation

    Returns:
        Array: The Cauchy stress tensor
    """

    F = get_F(fe, sol)
    Jinv = 1/np.linalg.det(F)
    P = get_P(sol, internal_vars, fe, P_func)
    return Jinv[:, :, None, None] * P @ onp.transpose(F, axes=(0, 1, 3, 2))

def surface_integral(sol: ArrayLike, internal_vars, fe: FiniteElement, location_fn: Callable, surface_int_fn: Callable) -> float:
    """Defines the surface integral to be used by compute_traction

    Args:
        fe (FiniteElement class): Contains mesh information for FE solve
        location_fn (function): describes the location where the surface integral
        is applied
        surface_int_fn (function): Computes the QoI on the surface,
        generally a traction give displacement solution
        sol (np.array): the flattened solution vector of displacements

    Returns:
        float: The value of the integral of surface_int_fn over location_fn
    """

    boundary_inds = fe.get_boundary_conditions_inds(location_fn)
    face_shape_grads_physical, nanson_scale = fe.get_face_shape_grads(boundary_inds)
    # (num_selected_faces, 1, num_nodes, vec, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
    cell_dofs = fe.local_to_cell_dofs(sol)[boundary_inds[:, 0]]
    u_grads_face = jax.vmap(calc_grads)(cell_dofs, face_shape_grads_physical)

    selected_face_shape_vals = fe.face_shape_vals[boundary_inds[:, 1]]
    int_vars_face = [internal_vars[var][boundary_inds[:, 0]] for var in internal_vars]
    int_vars_face = [np.sum(selected_face_shape_vals[:, :, :, None] * var[:, None, :, :], axis=2) for var in int_vars_face]

    normals = fe.get_surface_normals(location_fn)
    traction = surface_int_fn(u_grads_face, int_vars_face, normals) # (num_selected_faces, num_face_quads, vec)
    # (num_selected_faces, num_face_quads, vec) * (num_selected_faces, num_face_quads, 1)
    int_val = np.sum(traction * nanson_scale[:, :, None], axis=(0, 1))
    return int_val

def compute_traction(sol: ArrayLike, internal_vars, fe: FiniteElement, stress_fn: Callable, location_fn: Callable) -> float:
    """Computes the traction vector over a given surface

    Args:
        problem (Problem class): needed to obtain the tensor map for 
        the surface_fn
        fe (FiniteElement class): Contains mesh information for FE solve
        location_fn (function): describes the location where the surface integral
        sol (np.array): the flattened solution vector of displacements

    Returns:
        float: Traction value calculated over the location_fn
    """
    vmap_stress = jax.vmap(stress_fn)
    def traction_fn(u_grads, int_vars, normals):

        # (num_selected_faces, num_face_quads, vec, dim) -> (num_selected_faces*num_face_quads, vec, dim)
        u_grads_reshape = u_grads.reshape(-1, fe.vec, fe.dim)
        int_vars_reshape = [var.reshape(-1, *list(int_vars[0].shape)[2:]) for var in int_vars]
        sigmas = vmap_stress(u_grads_reshape, *int_vars_reshape).reshape(u_grads.shape)
        # (num_selected_faces, num_face_quads, vec, dim) @ (num_selected_faces, 1, dim, 1) -> (num_selected_faces, num_face_quads, vec, 1)
        traction = (sigmas @ normals[:, :, :, None])[:, :, :, 0]
        return traction

    traction_integral_val = surface_integral(sol, internal_vars, fe, location_fn, traction_fn)
    return traction_integral_val

#TODO: check this guy
def get_surface_normals_nodes_current(fe: FiniteElement, sol: ArrayLike, location_fn: Callable) -> ArrayLike:
    """ gets surface normals in the current configuration

    Args:
        u_fe_nodes (_type_): _description_
        bndry (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    bndry = fe.get_boundary_conditions_inds([location_fn])[0]    
    normals = fe.get_normals(bndry)[0]
    F = get_F(fe, sol)
    normals = np.einsum('bqij,bqj->bqi', F[bndry[:, 0]], normals)
    return normals/np.linalg.norm(normals, axis=-1)[:, :, None]
