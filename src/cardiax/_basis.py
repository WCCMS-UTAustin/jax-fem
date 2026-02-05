
"""
_basis.py is used for handling the basis functions and their properties.
The basis elements originate from the FEniCS basix library, with modification
that were done for JAX-FEM that we have kept. May want to see why it diverges from
basix at some point to see if we can more easily use the full library.
"""

import basix
import numpy as onp

from cardiax import logger

# Will be turning this into a class, and potentially creating a class instance and directly
# importing functions from the class, to allow for the re-use of shape function evaluation code. 
# This should allow for more flexibility with element type as well if desired.
#
# NOTE: this ^ might not be the best idea, I really am looking for a solution that allows me
#       to append to a dictionary...
#
# could also maybe partial get_face_shape_vals_and_grads or whatever other functions need to be used...

# maybe turn this into an abstract class?
class BasisFns:

    def __init__(self):
        """ initialization of data structures containing finite element types.
        """

        # TODO: turn this into a factory method and/or a custom data structure...
        self._element_dict = {'hexahedron': 
            {
                're_order': [0, 1, 3, 2, 4, 5, 7, 6],
                'element_family': basix.ElementFamily.P,
                'basix_ele': basix.CellType.hexahedron,
                'basix_face_ele': basix.CellType.quadrilateral,
                'gauss_order': 3,
                'degree': 1,
                'warning': None},
            'hexahedron27': {
                're_order': [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19,
                    17, 10, 12, 15, 14, 22, 23, 21, 24, 20, 25, 26],
                'element_family': basix.ElementFamily.P,
                'basix_ele': basix.CellType.hexahedron,
                'basix_face_ele': basix.CellType.quadrilateral,
                'gauss_order': 6, # 6x6x6, full integration
                'degree': 2 
            },
            'hexahedron20':
                {'re_order': [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19,
                            17, 10, 12, 15, 14],
                'element_family': basix.ElementFamily.serendipity,
                'basix_ele': basix.CellType.hexahedron,
                'basix_face_ele': basix.CellType.quadrilateral,
                'gauss_order': 2, # 6x6x6, full integration
                'degree': 2},
            'hexahedron64': {
                #TODO: this re_order variable is not correct, we need to see the algorithm basix uses to order dofs and the algorithm meshio uses
                're_order': [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13,
                            14, 15, 16, 17, 19, 18, 22, 23, 20, 21, 24, 25,
                            26, 27, 28, 29, 31, 30, 32, 34, 35, 33, 36, 37,
                            39, 38, 40, 42, 43, 41, 44, 45, 47, 46, 49, 48,
                            50, 51, 52, 53, 55, 54, 56, 57, 59, 58, 60, 61,
                            63, 62],
                'element_family': basix.ElementFamily.P,
                'basix_ele': basix.CellType.hexahedron,
                'basix_face_ele': basix.CellType.quadrilateral,
                'gauss_order': 9,
                'degree': 3,
                'warning': "Warning: 64-node hexahedron is currently in development and is not guaranteed to be correct."
            },
            'tetrahedron':
            {
                're_order': [0, 1, 2, 3],
                'element_family': basix.ElementFamily.P,
                'basix_ele': basix.CellType.tetrahedron,
                'basix_face_ele': basix.CellType.triangle,
                'gauss_order': 0, # 1, full integration
                'degree': 1
            },
            'tetra10':
            {
                're_order': [0, 1, 2, 3, 9, 6, 8, 7, 5, 4],
                'element_family': basix.ElementFamily.P,
                'basix_ele': basix.CellType.tetrahedron,
                'basix_face_ele': basix.CellType.triangle,
                'gauss_order': 2, # 4, full integration
                'degree': 2
            },
            'quad': {
                're_order': [0, 1, 3, 2],
                'element_family': basix.ElementFamily.P,
                'basix_ele': basix.CellType.quadrilateral,
                'basix_face_ele': basix.CellType.interval,
                'gauss_order': 2,
                'degree': 1
            },
            'quad8': {
                're_order': [0, 1, 3, 2, 4, 6, 7, 5],
                'element_family': basix.ElementFamily.serendipity,
                'basix_ele': basix.CellType.quadrilateral,
                'basix_face_ele': basix.CellType.interval,
                'gauss_order': 2,
                'degree': 2
            },
            'triangle': {
                're_order': [0, 1, 2],
                'element_family': basix.ElementFamily.P,
                'basix_ele': basix.CellType.triangle,
                'basix_face_ele': basix.CellType.interval,
                'gauss_order': 0, # 1, full integration
                'degree': 1
            },
            'triangle6': {
                're_order': [0, 1, 2, 5, 3, 4],
                'element_family': basix.ElementFamily.P,
                'basix_ele': basix.CellType.triangle,
                'basix_face_ele': basix.CellType.interval,
                'gauss_order': 2, # 3, full integration
                'degree': 2
            }
        }

    def get_elements(self, ele_type: str):
        """Mesh node ordering is important.
        If the input mesh file is Gmsh .msh or Abaqus .inp, meshio would convert it to
        its own ordering. My experience shows that meshio ordering is the same as Abaqus.
        For example, for a 10-node tetrahedron element, the ordering of meshio is the following
        https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node33.html
        The troublesome thing is that basix has a different ordering. As shown below
        https://defelement.com/elements/lagrange.html
        The consequence is that we need to define this "re_order" variable to make sure the
        ordering is correct.
        """

        try:
            element_info = self._element_dict[ele_type]
        except:
            raise NotImplementedError(f"Element type {ele_type} was not found! Please check that you're importing \
                                      get_elements() from the right place, that you're spelling/identifying the element \
                                      correctly, or request to add it as an element type.")

        # not sure if we want to allow for exceptions here...
        element_family = element_info['element_family']
        re_order = element_info['re_order']
        basix_ele = element_info['basix_ele']
        basix_face_ele = element_info['basix_face_ele']
        gauss_order = element_info['gauss_order']
        degree = element_info['degree']
        try:
            warning_msg = element_info['warning']
            print(warning_msg)
        # don't need to do anything if there's no warning message.
        except:
            pass

        return element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order

    def reorder_inds(self, inds, re_order):
        """Add documentation

        Args:
            inds (_type_): _description_
            re_order (_type_): _description_

        Returns:
            _type_: _description_
        """

        new_inds = []
        for ind in inds.reshape(-1):
            new_inds.append(onp.argwhere(re_order == ind))
        new_inds = onp.array(new_inds).reshape(inds.shape)
        return new_inds

    def get_shape_vals_and_grads(self, ele_type, gauss_order=None):
        """ Returns shape function values, gradients, and weights.

        Calls :py:meth:`jax_fem.basis.get_elements` to initialize data structures
        used to evaluate shape functions and shape function gradients at quadrature
        points for the given element. Dimensions shown below correspond to otuput for
        a HEX8/linear hexahedral element.

        Returns
        -------
        shape_values: ndarray
            (num_quads, num_nodes)
        shape_grads_ref: ndarray
            (num_quads, num_nodes, dim)
        weights: ndarray
            (num_quads,)
        """
        element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = self.get_elements(ele_type)

        if gauss_order is None:
            gauss_order = gauss_order_default

        quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)
        if degree >2:
            element = basix.create_element(element_family, basix_ele, degree, lagrange_variant = basix.LagrangeVariant.equispaced)
        else:
            element = basix.create_element(element_family, basix_ele, degree)
        vals_and_grads = element.tabulate(1, quad_points)[:, :, re_order, :]
        #print(type(quad_points))
        shape_values = vals_and_grads[0, :, :, 0]
        shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
        logger.debug(f"ele_type = {ele_type}, quad_points.shape = (num_quads, dim) = {quad_points.shape}")
        return shape_values, shape_grads_ref, weights


    def get_face_shape_vals_and_grads(self, ele_type, gauss_order=None, rule=None):
        """TODO: Returns shape function values, gradients, weights, 
        normal vectors, and indices on/of each face of a given FE.


        Calls :py:meth:`jax_fem.basis.get_elements` to initialize data structures
        used to evaluate shape functions and shape function gradients at quadrature
        points for the given element. Dimensions shown below correspond to otuput for
        a HEX8/linear hexahedral element.

        Returns shape function information, normals, and corresponding index of each
        FACE of an element. Returned values are used by pressure and/or surface kernels.

        Returns
        -------
        face_shape_vals: ndarray
            (num_faces, num_face_quads, num_nodes)
        face_shape_grads_ref: ndarray
            (num_faces, num_face_quads, num_nodes, dim)
        face_weights: ndarray
            (num_faces, num_face_quads)
        face_normals:ndarray
            (num_faces, dim)
        face_inds: ndarray
            (num_faces, num_face_vertices)
        """
        # TODO: Returns shape function values, gradients, weights, 
        # normal vectors, and indices on/of each face of a given FE.

        element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = self.get_elements(ele_type)

        if gauss_order is None:
            gauss_order = gauss_order_default

        # TODO: Check if this is correct.
        # We should provide freedom for seperate gauss_order for volume integral and surface integral
        # Currently, they're using the same gauss_order!
        # breakpoint()
        if rule is None:
            points, weights = basix.make_quadrature(basix_face_ele, gauss_order)
        else:
            # list of rules, from the basix documentation:
            # 0: default
            # 1: gauss_jacobi
            # 2: gll
            # 3: xiao_gimbutas
            # https://docs.fenicsproject.org/basix/main/python/_autosummary/basix.html#basix.QuadratureType
            points, weights = basix.make_quadrature(basix_face_ele, gauss_order, rule)
        
        # might need to debug this, some values seem off for the GLL rule... (rule=2)
        map_degree = 1
        lagrange_map = basix.create_element(basix.ElementFamily.P, basix_face_ele, map_degree)
        values = lagrange_map.tabulate(0, points)[0, :, :, 0]
        vertices = basix.geometry(basix_ele)
        dim = len(vertices[0])
        facets = basix.cell.sub_entity_connectivity(basix_ele)[dim - 1]
        # Map face points
        # Reference: https://docs.fenicsproject.org/basix/main/python/demo/demo_facet_integral.py.html
        face_quad_points = []
        face_inds = []
        face_weights = []
        # breakpoint()
        for f, facet in enumerate(facets):
            mapped_points = []
            for i in range(len(points)):
                vals = values[i]
                mapped_point = onp.sum(vertices[facet[0]] * vals[:, None], axis=0)
                mapped_points.append(mapped_point)
            face_quad_points.append(mapped_points)
            face_inds.append(facet[0])
            jacobian = basix.cell.facet_jacobians(basix_ele)[f]
            if dim == 2:
                size_jacobian = onp.linalg.norm(jacobian)
            else:
                size_jacobian = onp.linalg.norm(onp.cross(jacobian[:, 0], jacobian[:, 1]))
            face_weights.append(weights*size_jacobian)
        face_quad_points = onp.stack(face_quad_points)
        face_weights = onp.stack(face_weights)

        face_normals = basix.cell.facet_outward_normals(basix_ele)
        face_inds = onp.array(face_inds)
        face_inds = self.reorder_inds(face_inds, re_order)
        num_faces, num_face_quads, dim = face_quad_points.shape
        if degree >2:
            element = basix.create_element(element_family, basix_ele, degree, lagrange_variant = basix.LagrangeVariant.equispaced)
        else:
            element = basix.create_element(element_family, basix_ele, degree)
        vals_and_grads = element.tabulate(1, face_quad_points.reshape(-1, dim))[:, :, re_order, :]
        face_shape_vals = vals_and_grads[0, :, :, 0].reshape(num_faces, num_face_quads, -1)
        face_shape_grads_ref = vals_and_grads[1:, :, :, 0].reshape(dim, num_faces, num_face_quads, -1)
        face_shape_grads_ref = onp.transpose(face_shape_grads_ref, axes=(1, 2, 3, 0))
        logger.debug(f"face_quad_points.shape = (num_faces, num_face_quads, dim) = {face_quad_points.shape}")
        # breakpoint()
        return face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds

    # functionality for contact: get ALL nodes on a specific element face.
    # Accessing only the vertices on each element face, as done in get_face_shape_vals_and_grads,
    # is enough to determine facet normals, but we need the local indices of the nodes on each face
    # in order to properly assign gauss weights to them.
    # TODO: create a unit test that checks if the quadrature weights associated with these nodes
    #       are actually assigned correctly!!!
    def get_face_node_inds(self, ele_type, deg):
        """ returns the nodes on each face of a specific element.

        Args:
            ele_type (_type_): _description_
            deg (_type_): _description_
        """

        # get the basix element from the element type that is prescribed in CARDIAX
        element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = self.get_elements(ele_type)

        # breakpoint()
        if deg > 2:
            # must pass the lagrange variant if the element degree is over 3.
            lagrange_map_cell = basix.create_element(basix.ElementFamily.P, basix_ele, deg, lagrange_variant = basix.LagrangeVariant.equispaced)
        else:
            lagrange_map_cell = basix.create_element(basix.ElementFamily.P, basix_ele, deg)

        # entity_closure_dofs returns the nodes that comprise the vertices, lines,
        # faces, and entirety of a cell.
        face_node_inds = lagrange_map_cell.entity_closure_dofs[2]

        face_node_inds_np = onp.zeros((len(face_node_inds), len(face_node_inds[0])), dtype=onp.int32)
        # use the re-ordering supplied for the element type
        for i, face_inds in enumerate(face_node_inds):
            # breakpoint()
            face_inds = onp.array(face_inds)
            face_inds = self.reorder_inds(face_inds, re_order)

            face_node_inds_np[i] = face_inds

        return face_node_inds_np
        


###################################################################
## instantiation of methods to remain backwards-compatible       ##
###################################################################
# class to import functions from
basis_fn_class = BasisFns()

# functions that will get imported!
get_shape_vals_and_grads = basis_fn_class.get_shape_vals_and_grads
get_face_shape_vals_and_grads = basis_fn_class.get_face_shape_vals_and_grads
get_face_node_inds = basis_fn_class.get_face_node_inds
get_elements = basis_fn_class.get_elements
###################################################################






# deprecated element types
# elif ele_type == 'hexahedron125':
#     print(f"Warning: 125-node hexahedron is currently in development and is not guaranteed to be correct.")
#     #TODO: this re_order variable is not correct, we need to see the algorithm basix uses to order dofs and the algorithm meshio uses
#     re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13,
#                 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 24, 23,
#                 29, 30, 31, 26, 27, 28, 32, 33, 34, 35, 36, 37,
#                 38, 39, 40, 43, 42, 41, 44, 50, 52, 46, 47, 51,
#                 49, 45, 48, 53, 55, 61, 59, 54, 58, 60, 56, 57,
#                 62, 68, 70, 64, 65, 69, 67, 63, 66, 71, 73, 79,
#                 77, 72, 76, 78, 74, 75, 82, 80, 86, 88, 81, 83,
#                 87, 85, 84, 89, 91, 97, 95, 90, 94, 96, 92, 93,
#                 98, 100, 106, 104, 116, 118, 124, 122, 99, 101,
#                 107, 103, 109, 105, 115, 113, 117, 119, 121, 123,
#                 102, 108, 110, 112, 114, 120, 111]
#     basix_ele = basix.CellType.hexahedron
#     basix_face_ele = basix.CellType.quadrilateral
#     gauss_order = 12
#     degree = 4
# elif ele_type == 'hexahedron216':
#     print(f"Warning: 216-node hexahedron is currently in development and is not guaranteed to be correct.")
#     #TODO: this re_order variable is not correct, we need to see the algorithm basix uses to order dofs and the algorithm meshio uses
#     re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13,
#                 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
#                 26, 27, 31, 30, 29, 28, 36, 37, 38, 39, 32, 33,
#                 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
#                 50, 51, 55, 54, 53, 52, 56, 68, 71, 59, 60, 64,
#                 69, 70, 67, 63, 58, 57, 61, 65, 66, 62, 72, 75,
#                 87, 84, 73, 74, 79, 83, 86, 85, 80, 76, 77, 78,
#                 82, 81, 88, 100, 103, 91, 92, 96, 101, 102, 99,
#                 95, 90, 89, 93, 97, 98, 94, 104, 107, 119, 116,
#                 105, 106, 111, 115, 118, 117, 112, 108, 109, 110,
#                 114, 113, 123, 120, 132, 135, 122, 121, 124, 128,
#                 133, 134, 131, 127, 126, 125, 129, 130, 136, 139,
#                 151, 148, 137, 138, 143, 147, 150, 149, 144, 140,
#                 141, 142, 146, 145, 152, 155, 167, 164, 200, 203,
#                 215, 212, 153, 154, 156, 160, 168, 184, 159, 163,
#                 171, 187, 166, 165, 183, 199, 180, 196, 201, 202,
#                 204, 208, 207, 211, 214, 213, 157, 161, 162, 158,
#                 169, 170, 186, 185, 172, 188, 192, 176, 175, 179,
#                 195, 191, 182, 181, 197, 198, 205, 206, 210, 209,
#                 173, 174, 178, 177, 189, 190, 194, 193]
#     basix_ele = basix.CellType.hexahedron
#     basix_face_ele = basix.CellType.quadrilateral
#     gauss_order = 15
#     degree = 5