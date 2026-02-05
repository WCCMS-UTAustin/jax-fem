
import numpy as onp
import jax
import jax.numpy as np
import jax.flatten_util
from dataclasses import dataclass, field
from typing import Callable, Optional, Union
import functools
from jaxtyping import ArrayLike

from cardiax.common import timeit
from cardiax import FiniteElement
from cardiax import logger
from cardiax._internal import MethodWrappingMeta

""" WARNING!
The switch from lists to dictionaries has not been verified for mixed formulations.
PROCEED WITH CAUTION.

Also, as FYI:
The Problem class will also be abstracted to allow for more flexibility moving forward.
Not sure exactly how this will look yet, but the user should not notice anything after the
refactoring. The refactor will allow for customization of the Problem class through
overwriting specific methods to allow more flexibility and development of new features.
"""

@dataclass
class Problem(metaclass=MethodWrappingMeta):
    """This base class creates the objects needed to compute the linear system required to solve the PDE.
    This Problem class is used to create the initial FE space, boundary conditions, and necessary
    attributes for the computation.

    Args:
        fes (FiniteElement): Finite element object or list of finite element objects that define the problem.
        dirichlet_bc_info (Optional[Dict[Union[List[Callable], List[int], List[Callable]]]]): List of Dirichlet boundary condition information.
        location_fns (Optional[List[Callable]]): List of functions that define the locations for boundary conditions.
        fe_bindings (Optional[List[List[int]]]): List of finite element bindings, where each binding is a list of indices of finite elements that share degrees of freedom.
                                                useful for volumetric/deviatoric split with reduced integration
        additional_info (Any): Additional information that may be required for the problem setup.

    """

    fes: dict[str, FiniteElement]
    dirichlet_bc_info: Optional[dict[Union[list[Callable], list[int], list[Callable]]]] = None
    location_fns: Optional[dict[str, dict[str, Callable]]] = None
    fe_bindings: Optional[dict[str, int]] = None
    _allowed_to_set: bool = field(default=True, init=False, repr=False)

    def __post_init__(self):
        """
        Broke up the initialization in __post_init__ to happen in several steps for clarity.
        The order of operations is important, but there are places where we can consolidate.
        This should also make future development easier when we abstract Problem further.
        TODO: Will go through this after pull request is merged to clean up and comment more.
        Should potentially set self values here instead of inside the functions for clarity,
        not sure though.
        """
        
        self._allowed_to_set = True  # Allow setting attributes within the method
        self.basic_checks()
        self.setup_variables()
        self.determine_indices()
        self.determine_indices_face()
        self.initialize_fe_vars()
        self.update_dirichlet_info()
        self.pre_jit_fns()
        self._enforce_setattr = True  # Start enforcing after post-init
        self._allowed_to_set = False  # Allow setting attributes within the method

    def basic_checks(self):
        #Check to see if inputs are singletons, and if so, put them in a single element list
        assert isinstance(self.fes, dict)

        # TODO: Find out what to do about the No dirichlet condition later
        # this may work
        if self.dirichlet_bc_info is None:
            self.dirichlet_bc_info = {}
            for key in self.fes.keys():
                self.dirichlet_bc_info[key] = []
        return

    def setup_variables(self):
        #build lists of attributes for each finite element object
        self.mesh = {fe_key: self.fes[fe_key].mesh for fe_key in self.fes}
        self.vec = {fe_key: self.fes[fe_key].vec for fe_key in self.fes}
        self.dim = {fe_key: self.fes[fe_key].dim for fe_key in self.fes}
        self.ele_type = {fe_key: self.fes[fe_key].ele_type for fe_key in self.fes}
        self.gauss_order = {fe_key: self.fes[fe_key].gauss_order for fe_key in self.fes}
        self.support_list = {fe_key: self.fes[fe_key].get_cell_basis_supports() for fe_key in self.fes}
        self.num_cells = {fe_key: self.fes[fe_key].num_cells for fe_key in self.fes}
        return

    def determine_indices(self):
        #total number of finite element fields
        self.num_vars = len(self.fes)

        # FIXME: Will revisit this after code is working with dictionaries
        #build a list of finite element objects to exclude from the global dofs because they are bound (i.e. completely share dofs with another finite element object)
        #we do this so that we can perform reduced integration on volumetric energy penalties using two different finite element quadrature schemes but with the same
        #degrees of freedom used.
        global_dof_exclusions = []

        if self.fe_bindings is not None:
            for j, binding in enumerate(self.fe_bindings):

                binding = onp.sort(onp.unique(binding))
                self.fe_bindings[j] = binding
                bound_fe_num_total_dofs = [self.fes[i].num_total_dofs for i in binding]

                assert len(onp.unique(bound_fe_num_total_dofs)) == 1, "Bound finite element fields contain different numbers of degrees of freedom; cannot be bound together."

                exclusions = binding[1:]

                for x in onp.unique(exclusions):
                    global_dof_exclusions.append(x)

        global_dof_exclusions = onp.unique(global_dof_exclusions)
        # self.global_dof_inclusions = list(set(self.fes.keys()) - set(global_dof_exclusions))
        self.global_dof_inclusions = {key: self.fes[key] for key in self.fes if key not in global_dof_exclusions}

        #create offsets for global to local dof distribution
        # VERIFY THIS IS CORRECT WHEN CAPABLE OF DOING MULTI-VARIABLE
        self.offset = {}
        offset_list = [0]
        for i, key in enumerate(self.fes):
            if i == 0:
                self.offset[key] = 0
                continue
            if key in self.global_dof_inclusions:
                self.offset[key] = offset_list[-1] + self.fes[key].num_total_dofs
                offset_list.append(self.offset[key])

        self.local_to_global_dof_indices_vmap = jax.vmap(self.local_to_global_dofs_indices, in_axes=(0, None), out_axes = 0)
        self.local_cell_to_global_dof_indices_vmap = jax.vmap(jax.vmap(self.local_to_global_dofs_indices, in_axes=(0, None), out_axes = 0), in_axes = (0, None), out_axes = 0)

        # inds is an array with dimensions (num_cells, num dof per cell) with each row containing the indices 
        # of the global dofs with support on the corresponding cell. Because individual global dofs may have
        # support on multiple cells, the entries in the array are not in general unique.

        self.I = []
        self.J = []
        self.cell_inds = [] # might as well store this during initialization, need to use to determine free vs. fixed inds.
        self.internal_vars = {}

        for fe_index, key in enumerate(self.fes):
            inds = self.local_cell_to_global_dof_indices_vmap(self.fes[key].get_cell_dof_supports(), key)
            self.cell_inds.append(inds)
            indI = onp.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
            indJ = onp.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)
            self.I.append(indI)
            self.J.append(indJ)
            self.internal_vars[key] = {}

        self.I = onp.hstack(self.I)
        self.J = onp.hstack(self.J)
        return
    
    def determine_indices_face(self):
        if self.location_fns is not None:
            self.boundary_inds_dict = {}
            self.internal_vars_surfaces = {}
            for fe_key in self.fes:
                temp_boundary_inds = {}
                int_vars_surf = {}
                for key in self.location_fns[fe_key]:
                    boundary_inds = self.fes[fe_key].get_boundary_conditions_inds(self.location_fns[fe_key][key])
                    int_vars_surf[key] = {}
                    temp_boundary_inds[key] = boundary_inds
                self.boundary_inds_dict[fe_key] = temp_boundary_inds
                self.internal_vars_surfaces[fe_key] = int_vars_surf

        else:
            self.boundary_inds_dict = {fe_key: {} for fe_key in self.fes}
            self.internal_vars_surfaces = {fe_key: {} for fe_key in self.fes}

        self.cells_face_dict = {}

        for fe_key in self.fes:
            temp_face_dict = {} #np.empty((0, self.fes[fe_key].num_nodes * self.fes[fe_key].vec), dtype=np.int32)
            for bndry_key in self.boundary_inds_dict[fe_key]: #one for each finite element object
                bndry_inds = self.boundary_inds_dict[fe_key][bndry_key]
                cells_face = self.fes[fe_key].get_cell_dof_supports()[bndry_inds[:, 0]] # [(num_selected_faces, num_nodes), ...]
                inds_face  = self.local_cell_to_global_dof_indices_vmap(cells_face, fe_key)
                I_face = onp.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)
                J_face = onp.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)
                self.I = onp.hstack((self.I, I_face))
                self.J = onp.hstack((self.J, J_face))
                temp_face_dict[bndry_key] = cells_face
            self.cells_face_dict[fe_key] = temp_face_dict
        return

    def initialize_fe_vars(self):
        # This used to be looped over global_dof_inclusions
        # Should make dof_incs into dictionary as well to work with multiple finite element spaces
        self.num_total_dofs_all_vars = onp.sum(onp.array([self.fes[key].num_total_dofs for key in self.fes]))


        # each entry has dimensions (num_cells, num_quads)
        self.JxW = {key: fe.JxW for key, fe in self.fes.items()}
        #print(self.JxW[0].shape)
        # each entry has dimensions (num_cells, num_quads, num_nodes +..., dim)
        self.shape_grads = {key: fe.shape_grads for key, fe in self.fes.items()}
        # each entry has dimension (num_cells, num_quads, num_nodes + ..., 1, dim)
        self.v_grads_JxW = {key: fe.v_grads_JxW for key, fe in self.fes.items()}

        # each entry has dimension (num_cells, num_quads, num_nodes)
        self.shape_vals = {key: fe.cell_shape_vals for key, fe in self.fes.items()}


        # TODO: assert all vars quad points be the same
        # (num_cells, num_quads, dim)
        self.physical_quad_points = {key: fe.get_physical_quad_points() for key, fe in self.fes.items()}

        self.selected_face_shape_grads = {}
        self.nanson_scale = {}
        self.selected_face_shape_vals = {}
        self.physical_surface_quad_points = {}
        for fe_key in self.fes:
            s_shape_grads = {}
            n_scale = {}
            s_shape_vals = {}
            phys_quads = {}
            for bndry_key in self.boundary_inds_dict[fe_key]:
                boundary_inds = self.boundary_inds_dict[fe_key][bndry_key]
                # (num_selected_faces, num_face_quads, num_nodes, dim), (num_selected_faces, num_face_quads)
                face_shape_grads_physical, nanson_scale = self.fes[fe_key].get_face_shape_grads(boundary_inds)
                selected_face_shape_vals = self.fes[fe_key].face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
                physical_surface_quad_points = self.fes[fe_key].get_physical_surface_quad_points(boundary_inds)
                
                s_shape_grads[bndry_key] = face_shape_grads_physical
                n_scale[bndry_key] = nanson_scale
                s_shape_vals[bndry_key] = selected_face_shape_vals
                phys_quads[bndry_key] = physical_surface_quad_points
                #end loop

            self.selected_face_shape_grads[fe_key] = s_shape_grads
            self.nanson_scale[fe_key] = n_scale
            self.selected_face_shape_vals[fe_key] = s_shape_vals
            # TODO: assert all vars face quad points be the same
            # Shouldn't have to worry about same quads
            self.physical_surface_quad_points[fe_key] = phys_quads
        return
    
    def update_dirichlet_info(self):
        self.loc_to_glob_vmap = jax.vmap(self.local_to_global_dofs_indices, in_axes = (0, None))
        self.bc_inds = {}
        self.bc_vals = {}
        # May do this in a more user friendly way
        # Too many lists
        for key in self.fes:
            # TODO: allow this to be skipped if there are no dirichlet bc
            bc_inds = []
            bc_vals = []
            for bc in self.dirichlet_bc_info[key]:
                if len(self.dirichlet_bc_info[key]) > 0:
                    mark_fcs, components, value_fcs = bc
                    local_inds, vals = self.fes[key].get_dirichlet_data(mark_fcs, components, value_fcs)
                    bc_inds.append(onp.hstack(self.loc_to_glob_vmap(local_inds, key)))
                    bc_vals.append(onp.hstack(vals))
            
            try:
                self.bc_inds[key] = onp.hstack(bc_inds)
                self.bc_vals[key] = onp.hstack(bc_vals)
            except ValueError:
                self.bc_inds[key] = []
                self.bc_vals[key] = []

        # handling of problems with no dirichlet boundary conditions is required
        try:
            self.bc_inds = onp.hstack([x.flatten() for x in self.bc_inds.values()], dtype=onp.int32)
            self.bc_vals = onp.hstack([x.flatten() for x in self.bc_vals.values()])
        except AttributeError:
            self.bc_inds = np.array([self.num_total_dofs_all_vars + 1])
            self.bc_vals = np.array([0.])
        return

    def set_bc_vals(self, bc_vals):
        self.bc_vals = bc_vals
        return

    # This is a wrapper function to prevent the user from setting attributes directly
    # Intended to minimize accidental overwriting or incorrect naming of attributes
    def __setattr__(self, name, value):
        if getattr(self, '_enforce_setattr', False):
            if not getattr(self, '_allowed_to_set', False) and name not in ('_allowed_to_set', '_enforce_setattr'):
                raise AttributeError(f"Cannot add or modify attribute '{name}' directly. Use a designated method.")
        super().__setattr__(name, value)

    def custom_init(self):
        """Child class should override if more things need to be done in initialization
        """
        pass

    def get_laplace_kernel(self, 
                           tensor_map: Callable) -> Callable:
        """
        Used to compute the weak form derived from the laplacian.
        How do I math here?

        Args:
            tensor_map (callable): function that gives the \nabla u term for the weak form.
            Dimension should be d-D
            
        Returns:
            callable: function that computes the weak form of the laplacian
        """

        def laplace_kernel(cell_sol: ArrayLike, 
                           cell_shape_grads: ArrayLike, 
                           cell_v_grads_JxW: ArrayLike, 
                           *cell_internal_vars: ArrayLike) -> jax.Array:

            # cell_sol: (num_nodes, vec)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)


            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = np.einsum('ji,kjl->kil',cell_sol, cell_shape_grads)
            u_physics = jax.vmap(tensor_map)(u_grads, *cell_internal_vars)

            # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
            val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
            return val

        return laplace_kernel

    def get_mass_kernel(self, 
                        mass_map: Callable) -> Callable:
        """
        Used to compute the weak form derived from the mass/body force term.

        Args:
            mass_map (callable): function that gives the u term for the weak form.
            Dimension should be 1-D
            

        Returns:
            callable: function that computes the weak form of the mass term
        """

        def mass_kernel(cell_sol: ArrayLike, 
                        cell_shape_vals: ArrayLike, 
                        x: ArrayLike, 
                        cell_shape_grads: ArrayLike, 
                        cell_JxW: ArrayLike, 
                        *cell_internal_vars: ArrayLike) -> jax.Array:
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim) (quadrature points)
            # cell_JxW: (num_quads)

            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, num_nodes, vec) -> (num_quads, vec)
            u_grads = np.einsum('ji,kjl->kil',cell_sol, cell_shape_grads)
            u = np.sum(cell_sol[None, :, :] * cell_shape_vals[:, :, None], axis=1)
            u_physics = jax.vmap(mass_map)(u, u_grads, x, *cell_internal_vars)  # (num_quads, vec)
            
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * cell_shape_vals[:, :, None] * cell_JxW[:, None, None], axis=0)
            val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
            return val

        return mass_kernel

    def get_surface_kernel(self, 
                           surface_map: Callable) -> Callable:
        """
        Used to compute the weak form derived from the surface term.

        Args:
            surface_map (callable): function that gives the vector to be contracted with the normal for the weak form.
            Dimension should be d-D

        Returns:
            callable: function that computes the weak form of the surface term
        """

        def surface_kernel(cell_sol: ArrayLike, 
                           x: ArrayLike, 
                           face_shape_vals: ArrayLike, 
                           face_shape_grads: ArrayLike, 
                           face_nanson_scale: ArrayLike, 
                           *cell_internal_vars_surface: ArrayLike) -> jax.Array:
            # face_shape_vals: (num_face_quads, num_nodes + ...)
            # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
            # x: (num_face_quads, dim)
            # face_nanson_scale: (num_vars, num_face_quads)


            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = cell_sol[None, :, :, None] * face_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)
            u_physics = jax.vmap(surface_map)(u, u_grads, x, *cell_internal_vars_surface)  # (num_face_quads, vec)
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)

            return jax.flatten_util.ravel_pytree(val)[0]

        return surface_kernel

    def determine_free_index(self, 
                             fe_key: str) -> str:
        '''
        Determines the smallest index of a finite element field bound to self.fes[fe_index]
        '''
        if fe_key in self.global_dof_inclusions:
            free_fe_index = fe_key
        else:
            print("FE_bindings in Problem class should be None")
            print("We will add in the future")
            exit()
            # for bound in self.fe_bindings:
            #     if fe_index in bound:
            #         if bound[0] in self.global_dof_inclusions:
            #             free_fe_index = bound[0]
            #             break
        return free_fe_index

    def global_to_local_dofs(self, 
                             global_dofs: ArrayLike, 
                             fe_key: str) -> jax.Array:
        '''
        Accepts a flattened array of global degrees of freedom, and the index of a finite element
        field, and returns the local degrees of freedom for that finite element field.
        '''

        free_fe_index = self.determine_free_index(fe_key)
        dof_start_index = self.offset[free_fe_index]
        local_dofs = global_dofs[dof_start_index: dof_start_index + self.fes[free_fe_index].num_total_dofs]
        return local_dofs


    def local_to_global_dofs_indices(self, 
                                     local_dof, 
                                     fe_key):
        '''
        Accepts a local degree of freedom index, as well as a finite element field index, and returns
        the global degree of freedom index corresponding to that finite element field's local index
        '''
        free_fe_index = self.determine_free_index(fe_key)
        global_dof = local_dof + self.offset[free_fe_index]
        return global_dof

    def get_boundary_data(self):
        """
        dirichlet_dofs = []
        dirichlet_vals = []
        for fe_index, bc in enumerate(self.dirichlet_bc_info):
            mark_fcs, components, value_fcs = bc
            local_inds, vals = self.fes[fe_index].get_dirichlet_data(mark_fcs, components, value_fcs)
            dirichlet_dofs.append(self.loc_to_glob_vmap(local_inds, fe_index))
            dirichlet_vals.append(vals)

        dirichlet_dofs = onp.hstack(dirichlet_dofs, dtype = onp.int32)
        dirichlet_vals = onp.hstack(dirichlet_vals)
        return dirichlet_dofs, dirichlet_vals
        """
        return self.bc_inds, self.bc_vals
    
    def set_internal_vars(self, 
            internal_vars: dict[str, dict[str, ArrayLike]]) -> None:

        assert set(self.internal_vars.keys()) == set(internal_vars.keys()), \
            "Internal variable keys do not match finite element fields. \n" \
            f"{self.internal_vars.keys()} vs. {internal_vars.keys()}"
        
        int_var_temp = {}
        for fe_key in self.internal_vars:
            int_var = {}
            
            try:
                for var_key in internal_vars[fe_key]:
                    if len(internal_vars[fe_key][var_key]) == 0:
                        int_var[var_key] = ()
                    else:
                        var = internal_vars[fe_key][var_key]
                        # Cell data
                        if var.shape[0] == self.num_cells[fe_key]:
                            if var.shape[1] == self.fes[fe_key].num_quads:
                                var_reshaped = var
                            else:
                                if True: #TODO: Figure this out for tensor
                                    var_reshaped = np.repeat(var[:, None, :], repeats=self.fes[fe_key].num_quads, axis=1)
                                else:
                                    raise ValueError(f"Internal variable {var_key} for finite element "
                                                    f"field {fe_key} has incompatible shape {var.shape}."
                                                    f"Expected shape for cell data ({self.num_cells[fe_key]}, {self.fes[fe_key].num_quads}, {var} dim) or "
                                                    f"or ({self.num_cells[fe_key]}, {var} dim)")
                        # Nodal data
                        # elif (var.shape[0] == self.mesh[fe_key].points.shape[0]) | (var.shape[0] == self.mesh[fe_key].points.shape[0] * self.vec[fe_key]):
                        
                        elif (var.shape[0] == self.mesh[fe_key].points.shape[0]) | (var.shape[0] == self.mesh[fe_key].points.shape[0] * self.vec[fe_key]):
                            var_reshaped = self.fes[fe_key].convert_dof_to_quad(var)
                        # Constant data
                        elif sum(var.shape) == 1:
                            new_shape = (self.num_cells[fe_key], self.fes[fe_key].num_quads, *var.shape)
                            var_reshaped = np.full(new_shape, var, dtype=var.dtype)
                        else:
                            raise ValueError(f"Internal variable {var_key} for finite element "
                                            f"field {fe_key} has incompatible shape {var.shape}."
                                            f"Expected shape for nodal data ({self.mesh[fe_key].points.shape[0]}, {var} dim)")
                        int_var[var_key] = var_reshaped
            except Exception as e:
                raise ValueError(f"Error processing internal variable {var_key} for finite element field {fe_key}: {e}")
            int_var_temp[fe_key] = int_var
        
        self.internal_vars = int_var_temp
        return
    
    def set_internal_vars_surfaces(self, 
            internal_vars_surfaces: dict[str, dict[str, dict[str, ArrayLike]]]) -> None:
        
        assert set(self.internal_vars_surfaces.keys()) == set(internal_vars_surfaces.keys()), \
            "internal_vars_surfaces keys do not match finite element fields. \n" \
            f"{self.internal_vars_surfaces.keys()} vs. \n" \
            f"{internal_vars_surfaces.keys()}"

        int_var_surf_temp = {}
        for fe_key in self.internal_vars_surfaces:
            assert set(self.internal_vars_surfaces[fe_key].keys()) == set(internal_vars_surfaces[fe_key].keys()), \
            f"internal_vars_surfaces dict ({[fe_key]}) does not match surface functions. \n" \
            f"{self.internal_vars_surfaces[fe_key].keys()} vs. \n" \
            f"{internal_vars_surfaces[fe_key].keys()}"

            int_var_surf_fe = {}
            for surf_fn in self.internal_vars_surfaces[fe_key]:
                int_var_surf_fe_fn = {}
                try:
                    for var_key in internal_vars_surfaces[fe_key][surf_fn]:
                        var = internal_vars_surfaces[fe_key][surf_fn][var_key]
                        face_shape = self.physical_surface_quad_points[fe_key][surf_fn].shape
                        assert isinstance(var, jax.Array), \
                        f"Var {var_key} for surface function {surf_fn} in finite element field {fe_key} is not a jax array."
                        if len(var) == 0:
                            int_var_surf_fe_fn[var_key] = {}
                        else:
                            var = internal_vars_surfaces[fe_key][surf_fn][var_key]
                            # Cell face data
                            if var.shape[0] == face_shape[0]:
                                if len(var.shape) == 2:
                                    var_reshaped = np.repeat(var, repeats=self.fes[fe_key].num_quads, axis=1)                        
                                else:
                                    if var.shape[1] == face_shape[1]:
                                        var_reshaped = var
                                    else:
                                        raise ValueError(f"Internal variable {var_key} for finite element "
                                                        f"field {fe_key} has incompatible shape {var.shape}."
                                                        f"Expected shape for cell data ({self.num_cells[fe_key]}, {self.fes[fe_key].num_quads}, {var} dim) or "
                                                        f"or ({self.num_cells[fe_key]}, {var} dim)")
                            # Nodal face data
                            elif var.shape[0] == self.mesh[fe_key].points.shape[0]:
                                print("Nodal face data for internal vars on surfaces not implemented yet.")
                                exit()
                            # Constant face data
                            elif sum(var.shape) == 1:
                                var_reshaped = np.repeat(np.repeat(var[None, :], repeats=face_shape[1], axis=0)[None, :, :], repeats=face_shape[0], axis=0)
                            else:
                                raise ValueError(f"Internal variable {var_key} for finite element "
                                                f"field {fe_key} has incompatible shape {var.shape}."
                                                f"Expected shape for nodal data ({self.mesh[fe_key].points.shape[0]}, {var} dim)")
                            int_var_surf_fe_fn[var_key] = var_reshaped
                except Exception as e:
                    raise ValueError(f"Error processing self.internal_vars_surfaces for finite element field {fe_key}: {e}")
                int_var_surf_fe[surf_fn] = int_var_surf_fe_fn
            int_var_surf_temp[fe_key] = int_var_surf_fe

        self.internal_vars_surfaces = int_var_surf_temp
        return
    
    def set_dirichlet_bc_info(self, bc_info: dict[Union[list[Callable], list[int], list[Callable]]]) -> None:
        """
        Update the Dirichlet boundary condition information.

        Args:
            bc_info (dict): New Dirichlet boundary condition information.
        """
        self._enforce_setattr = False  # Allow setting attributes within the method
        self.dirichlet_bc_info = bc_info
        self.update_dirichlet_info()
        self._enforce_setattr = True  # Disallow setting attributes outside designated methods
        return

    def pre_jit_fns(self):
        def value_and_jacfwd(f, x):
            pushfwd = functools.partial(jax.jvp, f, (x, ))
            basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
            y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
            return y, jac

        def get_kernel_fn_cell():
            def kernel(cell_dofs, physical_quad_points, cell_shape_vals, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
                """
                universal_kernel should be able to cover all situations (including mass_kernel and laplace_kernel).
                mass_kernel and laplace_kernel are from legacy JAX-FEM. They can still be used, but not mandatory.
                """

                # TODO: If there is no kernel map, returning 0. is not a good choice. 
                # Return a zero array with proper shape will be better.
                if hasattr(self, 'get_mass_map'):
                    mass_kernel = self.get_mass_kernel(self.get_mass_map())
                    mass_val = mass_kernel(cell_dofs, cell_shape_vals, physical_quad_points, cell_shape_grads, cell_JxW, *cell_internal_vars)
                else:
                    mass_val = 0.

                if hasattr(self, 'get_tensor_map'):
                    laplace_kernel = self.get_laplace_kernel(self.get_tensor_map())
                    laplace_val = laplace_kernel(cell_dofs, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars)
                else:
                    laplace_val = 0.

                if hasattr(self, 'get_universal_kernel'):
                    universal_kernel = self.get_universal_kernel()
                    universal_val = universal_kernel(cell_dofs, physical_quad_points, cell_shape_grads, cell_JxW, 
                        cell_v_grads_JxW, *cell_internal_vars)
                else:
                    universal_val = 0.

                return laplace_val + mass_val + universal_val

            def kernel_jac(cell_dofs, *args):
                def kernel_partial(cell_dofs):
                    return kernel(cell_dofs, *args)
                return value_and_jacfwd(kernel_partial, cell_dofs)  # kernel(cell_dofs, *args), jax.jacfwd(kernel)(cell_dofs, *args)

            return kernel, kernel_jac

        def get_kernel_fn_face(key, ind):
            def kernel(cell_dofs, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
                """
                universal_kernel should be able to cover all situations (including surface_kernel).
                surface_kernel is from legacy JAX-FEM. It can still be used, but not mandatory.
                """

                if hasattr(self, 'get_surface_maps'):
                    surface_kernel = self.get_surface_kernel(self.get_surface_maps()[key][ind])
                    surface_val = surface_kernel(cell_dofs, physical_surface_quad_points, face_shape_vals,
                        face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
                else:
                    surface_val = 0.

                if hasattr(self, 'get_universal_kernels_surface'):
                    universal_kernel = self.get_universal_kernels_surface()[ind]
                    universal_val = universal_kernel(cell_dofs, physical_surface_quad_points, face_shape_vals,
                        face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
                else:
                    universal_val = 0.

                return surface_val + universal_val

            def kernel_jac(cell_dofs, *args):
                def kernel_partial(cell_dofs):
                    return kernel(cell_dofs, *args)
                return value_and_jacfwd(kernel_partial, cell_dofs)  # kernel(cell_dofs, *args), jax.jacfwd(kernel)(cell_dofs, *args)

            return kernel, kernel_jac

        # vmap each kernel over
        # Currently, this is only handled for 1 fe field
        # NOTE: This is what should be extended for mixed formulation
        kernel, kernel_jac = get_kernel_fn_cell()
        kernel = jax.jit(jax.vmap(kernel))
        kernel_jac = jax.jit(jax.vmap(kernel_jac))
        for fe_key in self.fes:
            self.kernel = {fe_key: kernel}
            self.kernel_jac = {fe_key: kernel_jac}

        ### TODO: Change up the surface checks now that we have dictionaries
        # count = [len(self.boundary_inds_dict[b]) for b in self.boundary_inds_dict]
        # num_surfaces = sum(count)
        # surfaces_check = 0
        # if hasattr(self, 'get_surface_maps'):
        #     surfaces_check += len(self.get_surface_maps())
        # elif hasattr(self, 'get_universal_kernels_surface'):
        #     surfaces_check += len(self.get_universal_kernels_surface())
        # else:
        #     assert num_surfaces == 0, "Missing definitions for surface integral"
        # print(f"num_surfaces: {num_surfaces}")
        # print(f"surfaces_check: {surfaces_check}")
        # assert num_surfaces == surfaces_check

        self.kernel_face = {}
        self.kernel_jac_face = {}
        for fe_key in self.fes:
            temp_face = {}
            temp_jac_face = {}
            for bndry_key in self.boundary_inds_dict[fe_key]:
                kernel_face, kernel_jac_face = get_kernel_fn_face(fe_key, bndry_key)
                kernel_face = jax.jit(jax.vmap(kernel_face))
                kernel_jac_face = jax.jit(jax.vmap(kernel_jac_face))
                temp_face[bndry_key] = kernel_face
                temp_jac_face[bndry_key] = kernel_jac_face
            self.kernel_face[fe_key] = temp_face
            self.kernel_jac_face[fe_key] = temp_jac_face

    @timeit
    def split_and_compute_cell(self, cells_dof_dict: dict[str, ArrayLike], 
                               jac_flag: bool, 
                               internal_vars: dict) -> Union[dict, tuple[dict, dict]]:
        """Volume integral in weak form
        #TODO: Can see if we can make using jax.lax.for loops and can possibly jit
        the entire function, not just the kernel_fns

        inputs:
            cells_dof_list: a list with one entry per finite element field. Each entry is an array of shape
            (num_cells, num_bases, vec) containing the finite element cell degrees of freedom per for each cell
            in the finite element field.

            jac_flag: boolean determining if we want to jacobian values as well

            internal_vars: remaining internal variables
        """
        fe_values = {}
        fe_jacs = {}
        for fe_key in self.fes:
            vmap_kernel_fn = self.kernel_jac[fe_key] if jac_flag else self.kernel[fe_key]
            values = []
            jacs = []
            input_collection = [cells_dof_dict[fe_key], self.physical_quad_points[fe_key], self.shape_vals[fe_key], self.shape_grads[fe_key], self.JxW[fe_key], self.v_grads_JxW[fe_key], *list(internal_vars[fe_key].values())]
            num_cuts = 1 # Number of cuts should be static, and changed by user
            # TODO: Investigate the effect of num_cuts if it makes 
            # a large difference from a memory perspective (should be faster with less)
            batch_size = self.num_cells[fe_key] // num_cuts

            for i in range(num_cuts):
                if i < num_cuts - 1:
                    input_col = jax.tree.map(lambda x: x[i * batch_size:(i + 1) * batch_size], input_collection, is_leaf = lambda y: isinstance(y, type((1,))))
                else:
                    input_col = jax.tree.map(lambda x: x[i * batch_size:], input_collection, is_leaf = lambda y: isinstance(y, type((1,))))

                val = vmap_kernel_fn(*input_col)
                if jac_flag:
                    values.append(val[0])
                    jacs.append(val[1])
                else:
                    values.append(val)

            values = np.vstack(values)
            fe_values[fe_key] = values.reshape(-1)
            if jac_flag:
                jacs = np.vstack(jacs)
                fe_jacs[fe_key] = jacs.reshape(-1)

        if jac_flag:
            return fe_values, fe_jacs
        else:
            return fe_values


    @timeit
    def compute_face(self, cells_dof_list: dict, jac_flag: bool, 
                     internal_vars_surfaces: Union[tuple, dict]) -> Union[jax.Array, tuple[jax.Array, jax.Array]]:
        """Surface integral in weak form
        #TODO: Similar to split_and_compute_cell, can see if we can make using jax.lax.for loops and can possibly jit
        #TODO: Should be a way to remove the if else statement by choosing jac_flag or not outside of this function
        """
        if jac_flag:
            values = {}
            jacs = {}
            for fe_key in self.fes:
                # fe_vals = np.empty((0, self.fes[fe_key].num_nodes * self.fes[fe_key].vec), dtype=np.int32)
                # fe_jacs = np.empty((0, self.fes[fe_key].num_nodes * self.fes[fe_key].vec, self.fes[fe_key].num_nodes * self.fes[fe_key].vec), dtype=np.int32)
                fe_vals = {}
                fe_jacs = {}
                for bndry_key in self.boundary_inds_dict[fe_key]:
                    vmap_fn = self.kernel_jac_face[fe_key][bndry_key]
                    bndry_inds = self.boundary_inds_dict[fe_key][bndry_key]
                    selected_cell_sols_flat = cells_dof_list[fe_key][bndry_inds[:, 0]]  # (num_selected_faces, num_nodes*vec + ...))

                    # May need to add cell # and fe index like in split and compute

                    input_collection = [selected_cell_sols_flat, self.physical_surface_quad_points[fe_key][bndry_key],
                                        self.selected_face_shape_vals[fe_key][bndry_key], self.selected_face_shape_grads[fe_key][bndry_key],
                                        self.nanson_scale[fe_key][bndry_key], *list(internal_vars_surfaces[fe_key][bndry_key].values())]
                    # Fix internal vars surface
                    val, jac = vmap_fn(*input_collection)
                    fe_vals[bndry_key] = val
                    fe_jacs[bndry_key] = jac
                values[fe_key] = fe_vals
                jacs[fe_key] = fe_jacs
            return values, jacs
        else:
            values = {}
            for fe_key in self.fes:
                # fe_vals = np.empty((0, self.fes[fe_key].num_nodes * self.fes[fe_key].vec), dtype=np.int32)
                fe_vals = {}
                for bndry_key in self.boundary_inds_dict[fe_key]:
                    vmap_fn = self.kernel_face[fe_key][bndry_key]
                    bndry_inds = self.boundary_inds_dict[fe_key][bndry_key]
                    selected_cell_sols_flat = cells_dof_list[fe_key][bndry_inds[:, 0]]  # (num_selected_faces, num_nodes*vec + ...))
                    # TODO: duplicated code
                    input_collection = [selected_cell_sols_flat, self.physical_surface_quad_points[fe_key][bndry_key],
                                        self.selected_face_shape_vals[fe_key][bndry_key], self.selected_face_shape_grads[fe_key][bndry_key],
                                        self.nanson_scale[fe_key][bndry_key], *list(internal_vars_surfaces[fe_key][bndry_key].values())]
                    # inspect val/the actual value returned by the integrals in the surface kernel
                    val = vmap_fn(*input_collection)
                    fe_vals[bndry_key] = val
                values[fe_key] = fe_vals
            return values

    def compute_residual_vars_helper(self, weak_form: ArrayLike, weak_form_face: ArrayLike) -> jax.Array:
        r''' computes residual vector

        computes residual vector $f$ in $Au=f$ by summing the contributions of the weak form appropriately.
        No sum occurs in the traditional FE setting, where weak form entries just get mapped to global res_vec
        entries. In the IGA setting, dofs defined at lagrange points are summed to determine their contribution
        to a given dof in IGA space, $$u^{\text{iga}}_j = \sum_{i=0}^N u^{\text{fe}}_{ij}$$ where the first index $i$
        represents the $i^{\text{th}}$ dof in IGA space, while the second index $j$ represents a mask that identifies
        the FE dof that contribute to the $i^{\text{th}}$ IGA dof. This mask is defined by :py:obj:`jax\_fem.splines.BSpline.support_list`.

        Parameters
        ----------
        weak_form_flat : np.ndarray
            weak form evaluated at lagrange nodes; test function is factored out.
        weak_form_face_flat : np.ndarray
            weak form evaluated at lagrange nodes on faces of the domain; test function is factored out.

        Returns
        -------
        np.ndarray
            residual vector to be used to solve linear system of equations $Au=f$, where $A$ is the stiffness mat.
        '''
        res_dict = {fe_key: np.zeros((self.fes[fe_key].num_total_nodes, 
                              self.fes[fe_key].vec)) for fe_key in self.fes}

        res_dict = {fe_key: res_dict[fe_key].at[self.support_list[fe_key].reshape(-1)].add(weak_form[fe_key].reshape(-1,
            self.fes[fe_key].vec)) for i, fe_key in enumerate(self.fes)}

        # res_dict = {fe_key: res_dict[fe_key].reshape(-1).at[self.cells_face_dict[fe_key][bndry_key].reshape(-1)].add(weak_form_face[fe_key][bndry_key].reshape(-1)) \
        #             for fe_key in self.fes for bndry_key in self.boundary_inds_dict[fe_key]}

        for fe_key in self.fes:
            a = jax.flatten_util.ravel_pytree(self.cells_face_dict[fe_key])[0].astype(int)
            b = jax.flatten_util.ravel_pytree(weak_form_face[fe_key])[0]
            res_dict[fe_key] = res_dict[fe_key].reshape(-1).at[a].add(b)

        res_vec = np.zeros((self.num_total_dofs_all_vars,))
        for fe_key in self.fes:
            res = res_dict[fe_key]
            res_vec = res_vec.at[self.loc_to_glob_vmap(np.array(range(len(res))), fe_key)].add(res)

        return res_vec

    def compute_residual_helper(self, global_dofs: ArrayLike, internal_vars: dict, 
                                internal_vars_surfaces: dict) -> jax.Array:
        """Computes the residual for each finite element field by passing the weak forms to the helper function

        NOTE: Needs to be setup like this where params to invert 
            (ie internal and internal_surface variables) are functional,
            Otherwise, run into tracer errors when solving inverse problems

        Args:
            global_dofs (_type_): _description_
            internal_vars (_type_): _description_
            internal_vars_surfaces (_type_): _description_

        Returns:
            _type_: _description_
        """
        logger.debug("Computing cell residual...")
        cells_dof_dict = {key: self.fes[key].local_to_cell_dofs(self.global_to_local_dofs(global_dofs, key)) for key in self.fes}
        #each entry in cells_sol_list has shape (num_cells,num_bases_per_cell,vec)
        weak_form_flat = self.split_and_compute_cell(cells_dof_dict, False, internal_vars)
        weak_form_face_flat = self.compute_face(cells_dof_dict, False, internal_vars_surfaces)  # [(num_selected_faces, num_nodes*vec + ...), ...]
       
        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

    def compute_residual(self, global_dofs: ArrayLike) -> jax.Array:
        """A partialed version of compute_residual_helper
        May not be necessary if jax.grad correctly

        """
        return self.compute_residual_helper(global_dofs, self.internal_vars, self.internal_vars_surfaces)

    def newton_update_helper(self, global_dofs: ArrayLike, internal_vars: dict, 
                             internal_vars_surfaces: dict) -> tuple[jax.Array, jax.Array]:
        logger.debug("Computing cell Jacobian and cell residual...")

        # goal: update global_to_local_dofs to map FREE dofs to their corresponding dofs on a cell level.
        #       split_and_compute_cell uses local/cell dofs
        cells_dof_dict = {key: self.fes[key].local_to_cell_dofs(self.global_to_local_dofs(global_dofs, key)) for key in self.fes}
        
        # the repeated dofs that we're saving are local dofs I think... this might need to get updated eventually.
        
        #each entry in cells_sol_list has shape (num_cells,num_bases_per_cell,vec)
        weak_form, cells_jac = self.split_and_compute_cell(cells_dof_dict, True, internal_vars)

        V = jax.flatten_util.ravel_pytree(cells_jac)[0]

        # [(num_selected_faces, num_nodes*vec + ...,), ...], [(num_selected_faces, num_nodes*vec + ..., num_nodes*vec + ...,), ...]
        weak_form_face, cells_jac_face = self.compute_face(cells_dof_dict, True, internal_vars_surfaces)
        
        # By default, returns an array of shape (0,) if there are no surface integrals
        V_face_flat = jax.flatten_util.ravel_pytree(cells_jac_face)[0]
        if V_face_flat.size == 1:
            pass
        else:
            V = np.hstack((V, jax.flatten_util.ravel_pytree(cells_jac_face)[0]))

        # changing what is passed to allow for the reduction of the system!
        # return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat), V

        # print(V.shape)
        # print(self.red_idx_vec.shape)

        res_vec = self.compute_residual_vars_helper(weak_form, weak_form_face)

        return res_vec, V

    def newton_update(self, global_dofs: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """
        Similar to compute_residual where it is a partialed version of newton_update_helper.
        Also may not be necessary
        """

        return self.newton_update_helper(global_dofs, self.internal_vars, self.internal_vars_surfaces)

    # this function is used to help handle integrals with time-dependent quantities.
    def u_global_to_u_cell_quads(self, u_global, fe_index):
        """ maps global displacements to quadrature points.

        accepts a displacement field and returns displacements of each cell at quadrature
        points;

        (total dofs,) -> (num_cells, num_quads, vec)

        Args:
            u_global (_type_): _description_
        """

        # NOTE: I don't really know the best behavior for this in the context of multiple FE fields...
        u = u_global.reshape(self.fes[fe_index].nodes.shape) # this is fragile and probably only works if these is 1 FE field...

        # convert to solution at nodes of each cell
        u_cell = u[self.fes[fe_index].cells]

        # compute things at quadrature points
        u_quads = onp.einsum('ijk, lj -> ilk', u_cell, self.fes[fe_index].shape_vals)

        return u_quads
        
    def u_global_to_u_cell(self, u_global, fe_index):
        """ maps global displacements to quadrature points.

        accepts a displacement field and returns displacements of each cell at quadrature
        points;

        (total dofs,) -> (num_cells, num_quads, vec)

        Args:
            u_global (_type_): _description_
        """

        # NOTE: need to check which FE function spaces this works for...
        # u = u_global.reshape(self.fes[fe_index].nodes.shape) # this is fragile and probably only works if these is 1 FE field...

        u = u_global.reshape(self.fes[fe_index].points.shape)

        # convert to solution at nodes of each cell
        u_cell = u[self.fes[fe_index].cells]

        return u_cell

    def set_params(self, params):
        """Used for solving inverse problems.
        """
        raise NotImplementedError("Child class must implement this function!")