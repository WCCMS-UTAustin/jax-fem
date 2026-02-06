import numpy as onp
import meshio
from meshio import Mesh
import pyvista as pv
import os
import gmsh
from typing import Optional
from jax.typing import ArrayLike

from cardiax._basis import get_elements

def gmsh_to_meshio(msh_file: Optional[str] = None, **kwargs) -> Mesh:
    """Converts gmsh output to meshio format.
    Currently deletes all mesh data, will change to 
    keep gmsh QoIs.

    Args:
        msh_file (Union[str, None]): The output msh file path.

    Returns:
        Mesh.meshio: The converted mesh object.
    """

    mesh = meshio.read("temp.msh")
    mesh.cell_data = kwargs.get("cell_data", {})
    mesh.point_data = kwargs.get("point_data", {})
    mesh.cell_sets = {}
    main_type = mesh.cells[-1].type
    temp = []
    [temp.append(c) for c in mesh.cells if c.type == main_type]
    if len(temp) > 1:
        combined_data = onp.concatenate([c.data for c in temp])
        temp = [meshio.CellBlock(cell_type=main_type, data=combined_data)]
    mesh.cells = temp
    if isinstance(msh_file, str):
        mesh.write(msh_file)

    os.remove("temp.msh")
    try:
        pvmesh = pv.from_meshio(mesh)
    except:
        raise RuntimeError("pyvista conversion failed. Please fix gmsh_to_meshio conversion" \
        " for your mesh.")

    return pvmesh

def rectangle_mesh(Nx: int = 10, Ny: int = 10, 
                   Lx: float = 1.0, Ly: float = 1.0, 
                   ele_type: str = "quad", 
                   msh_file: Optional[str] = None, 
                   verbose: bool = False) -> Mesh:
    """Generate a mesh for a rectangle using gmsh.
    
    Args:
        Nx (int): Number of elements in x direction
        Ny (int): Number of elements in y direction  
        Lx (float): Length in x direction
        Ly (float): Length in y direction
        degree (int): Degree of mesh elements
        data_dir (str): Directory to save mesh files if desired
        verbose (bool): Enable/disable gmsh terminal output
        
    Returns:
        meshio.Mesh: Mesh object
    """
    
    offset_x = 0.
    offset_y = 0.
    domain_x = Lx
    domain_y = Ly

    _, _, _, _, degree, _ = get_elements(ele_type)
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # save in old MSH format

    # Need to figure out if quad8 or quad9, below is for quad8
    # Configure to get quad8 (serendipity) instead of quad9 (complete)
    if ele_type == "quad8":
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)

    # Create rectangle geometry
    p1 = gmsh.model.geo.addPoint(offset_x, offset_y, 0)
    p2 = gmsh.model.geo.addPoint(offset_x + domain_x, offset_y, 0)
    p3 = gmsh.model.geo.addPoint(offset_x + domain_x, offset_y + domain_y, 0)
    p4 = gmsh.model.geo.addPoint(offset_x, offset_y + domain_y, 0)
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    
    # Set transfinite lines for structured mesh
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, Nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, Ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, Nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, Ny + 1)
    
    # Set transfinite surface and recombine to get quads
    if "quad" in ele_type:
        gmsh.model.geo.mesh.setTransfiniteSurface(s)
        gmsh.model.geo.mesh.setRecombine(2, s)
    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(degree)
    
    gmsh.write("temp.msh")

    mesh = gmsh_to_meshio(msh_file)
    mesh.points = mesh.points[:, :2]  # Drop z-coordinate for 2D mesh
    return mesh

def box_mesh(Nx: int = 10, Ny: int = 10, Nz: int = 10, 
             Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0, 
             ele_type: str = 'hexahedron', 
             msh_file: Optional[str] = None, 
             verbose: bool = False) -> Mesh:
    """
    Generate a structured box mesh using gmsh.geo API.
    """

    _, _, _, _, degree, _ = get_elements(ele_type)
    
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    if ele_type == 'hexahedron20':
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)

    # Create 4 corner points for the base rectangle
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0)

    # Create lines for the rectangle
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Create curve loop and surface
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])

    # Set transfinite lines for structured mesh
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, Nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, Nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, Ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, Ny + 1)

    # Set transfinite surface and recombine to get quads
    gmsh.model.geo.mesh.setTransfiniteSurface(s)
    gmsh.model.geo.mesh.setRecombine(2, s)

    # Extrude the surface in z to create the volume
    extruded = gmsh.model.geo.extrude([(2, s)], 0, 0, Lz, [Nz], [1], recombine=True)
    # The volume entity is the one with dim==3
    vol = next(tag for dim, tag in extruded if dim == 3)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = gmsh_to_meshio(msh_file)
    return mesh

def sphere_mesh(center: onp.ndarray = onp.array([0., 0., 0.]), 
                radius: float = 1.0, 
                degree: int = 1, 
                msh_file: Optional[str] = None, 
                verbose: bool = False) -> Mesh:

    # Initialize gmsh
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    # Create a new model
    gmsh.model.add("sphere")

    # Create a sphere
    # Parameters: center coordinates (x, y, z), radius
    center_x, center_y, center_z = center

    # Add a sphere volume
    gmsh.model.occ.addSphere(center_x, center_y, center_z, radius)

    # Synchronize the CAD representation with the gmsh model
    gmsh.model.occ.synchronize()

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)
    gmsh.option.setNumber("Mesh.ElementOrder", degree)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Write mesh to file
    gmsh.write("temp.msh")

    # Finalize gmsh
    gmsh.finalize()

    mesh = gmsh_to_meshio(msh_file)
    return mesh

def hollow_sphere_mesh(center: onp.ndarray = onp.array([0., 0., 0.]), 
                       outer_radius: float = 1.0, 
                       inner_radius: float = 0.5, 
                       degree: int = 1, 
                       msh_file: Optional[str] = None, 
                       verbose: bool = False) -> Mesh:
    """
    Create a hollow sphere by subtracting an inner sphere from an outer sphere.
    
    Args:
        center: tuple of (x, y, z) coordinates for sphere center
        outer_radius: radius of the outer sphere
        inner_radius: radius of the inner sphere (must be < outer_radius)
        degree: mesh element order
        verbose: Enable/disable gmsh terminal output
    
    Returns:
        mesh: meshio mesh object
    """
    
    # Initialize gmsh
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    # Create a new model
    gmsh.model.add("hollow_sphere")

    # Extract center coordinates
    center_x, center_y, center_z = center

    # Add outer sphere volume
    outer_sphere_tag = gmsh.model.occ.addSphere(center_x, center_y, center_z, outer_radius)
    
    # Add inner sphere volume
    inner_sphere_tag = gmsh.model.occ.addSphere(center_x, center_y, center_z, inner_radius)

    # Subtract inner sphere from outer sphere to create hollow sphere
    gmsh.model.occ.cut([(3, outer_sphere_tag)], [(3, inner_sphere_tag)])

    # Synchronize the CAD representation with the gmsh model
    gmsh.model.occ.synchronize()

    # Set mesh size based on the thickness of the shell
    shell_thickness = outer_radius - inner_radius
    char_length_min = shell_thickness / 10.0
    char_length_max = shell_thickness / 5.0
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length_max)
    gmsh.option.setNumber("Mesh.ElementOrder", degree)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Write mesh to file
    gmsh.write("temp.msh")

    # Finalize gmsh
    gmsh.finalize()

    mesh = gmsh_to_meshio(msh_file)
    return mesh

def ellipsoid_mesh(a: float = 1.0, b: float = 0.8, c: float = 0.6, 
                   mesh_size: float = 0.1, 
                   msh_file: Optional[str] = None, 
                   verbose: bool = False) -> Mesh:
    """
    Create an ellipsoid mesh using gmsh
    
    Parameters:
    a, b, c: semi-axes lengths (default creates a cardiac-like shape)
    mesh_size: characteristic mesh size
    output_file: output mesh file name
    verbose: Enable/disable gmsh terminal output
    """
    # Initialize gmsh
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ellipsoid")
    
    # Create ellipsoid geometry
    # gmsh uses a sphere and then scales it to create an ellipsoid
    sphere_tag = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
    
    # Scale the sphere to create ellipsoid with semi-axes a, b, c
    gmsh.model.occ.dilate([(3, sphere_tag)], 0, 0, 0, a, b, c)
    
    # Synchronize to update the geometry
    gmsh.model.occ.synchronize()
    
    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    
    # Generate 3D mesh
    gmsh.model.mesh.generate(3)
    
    # Write mesh file
    gmsh.write("temp.msh")
    
    # Optional: launch GUI to visualize (comment out for batch processing)
    # gmsh.fltk.run()
    
    # Finalize gmsh
    gmsh.finalize()

    mesh = gmsh_to_meshio(msh_file)
    return mesh

def hollow_ellipsoid_mesh(center: onp.ndarray = onp.array([0., 0., 0.]), 
                          outer_axes: ArrayLike = (1.0, 0.8, 0.6), 
                          inner_axes: ArrayLike = (0.5, 0.4, 0.3), 
                          degree: int = 1, 
                          cl_min: float = 0.05, 
                          cl_max: float = 0.1,
                          msh_file: Optional[str] = None,
                          verbose=False) -> Mesh:
    """
    Create a hollow ellipsoid by subtracting an inner ellipsoid from an outer ellipsoid.

    Args:
        center: tuple or array of (x, y, z) coordinates for ellipsoid center
        outer_axes: tuple of (a, b, c) semi-axes for outer ellipsoid
        inner_axes: tuple of (a, b, c) semi-axes for inner ellipsoid (must be < outer_axes)
        degree: mesh element order
        verbose: Enable/disable gmsh terminal output

    Returns:
        mesh: meshio mesh object
    """

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("hollow_ellipsoid")

    center_x, center_y, center_z = center

    # Add outer sphere and scale to ellipsoid
    outer_sphere_tag = gmsh.model.occ.addSphere(center_x, center_y, center_z, 1.0)
    gmsh.model.occ.dilate([(3, outer_sphere_tag)], center_x, center_y, center_z, *outer_axes)

    # Add inner sphere and scale to ellipsoid
    inner_sphere_tag = gmsh.model.occ.addSphere(center_x, center_y, center_z, 1.0)
    gmsh.model.occ.dilate([(3, inner_sphere_tag)], center_x, center_y, center_z, *inner_axes)

    # Subtract inner ellipsoid from outer ellipsoid
    gmsh.model.occ.cut([(3, outer_sphere_tag)], [(3, inner_sphere_tag)])

    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)
    gmsh.option.setNumber("Mesh.ElementOrder", degree)

    gmsh.model.mesh.generate(3)
    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = gmsh_to_meshio(msh_file)
    return mesh

def cylinder_mesh(height=1.0, 
                  radius=0.25, 
                  element_degree=1,
                  cl_min=0.1, 
                  cl_max=0.2, 
                  msh_file=None, 
                  verbose=False):
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cylinder")

    # Cylinder axis: z direction
    axis = [0, 0, height]
    gmsh.model.occ.addCylinder(0, 0, 0, axis[0], axis[1], axis[2], radius, tag=1)
    gmsh.model.occ.synchronize()

    # Mesh options
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)
    gmsh.option.setNumber("Mesh.ElementOrder", element_degree)
    gmsh.model.mesh.generate(3)

    # Save mesh
    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = gmsh_to_meshio(msh_file)
    return mesh

def hollow_cylinder_mesh(height: float = 1.0, 
                         outer_radius: float = 0.25, 
                         inner_radius: float = 0.1, 
                         element_degree: int = 1,
                         cl_min: float = 0.1, 
                         cl_max: float = 0.2, 
                         msh_file: Optional[str] = None, 
                         verbose: bool = False):
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("hollow_cylinder")

    # Outer cylinder
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius, tag=1)
    # Inner cylinder (to subtract)
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, inner_radius, tag=2)
    gmsh.model.occ.cut([(3, 1)], [(3, 2)], tag=3)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)
    gmsh.option.setNumber("Mesh.ElementOrder", element_degree)
    gmsh.model.mesh.generate(3)

    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = gmsh_to_meshio(msh_file)
    return mesh

def prolate_spheroid_mesh(msh_file: Optional[str] = None, 
                          verbose: bool = False,
                          cl_min: float = 0.25,
                          cl_max: float = 0.5) -> Mesh:

    sigma_min, sigma_max = 1.35, 1.8
    tau_min, tau_max = -1., 0.
    phi_min = 0.

    def compute_foci(tau):
        return onp.sqrt(3 * (1 + 5. * tau**2))

    center_pt = onp.array([0.0, 0.0, 0.0])
    axis_rotation = onp.array([0.0, 0.0, -1.0])

    def PS_coords(x):
        sigma, tau, phi = x
        x0 = compute_foci(tau) * onp.sqrt((sigma**2 - 1) * (1 - tau**2)) * onp.cos(phi)
        x1 = compute_foci(tau) * onp.sqrt((sigma**2 - 1) * (1 - tau**2)) * onp.sin(phi)
        x2 = compute_foci(tau) * sigma * tau
        return onp.array([x0, x1, x2])

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Geometry.CopyMeshingMethod", 1)
    gmsh.model.add("prolate_spheroid")

    apex_endo_pt = PS_coords(onp.array([sigma_min, tau_min, phi_min]))
    apex_epi_pt = PS_coords(onp.array([sigma_max, tau_min, phi_min]))
    apex_endo = gmsh.model.geo.addPoint(*apex_endo_pt)
    apex_epi = gmsh.model.geo.addPoint(*apex_epi_pt)
    center = gmsh.model.geo.addPoint(*center_pt)
    apex = gmsh.model.geo.addLine(apex_endo, apex_epi)

    for i in range(2):
        base_endo_pt = PS_coords(onp.array([sigma_min, tau_max, phi_min + i * onp.pi]))
        base_epi_pt = PS_coords(onp.array([sigma_max, tau_max, phi_min + i * onp.pi]))

        base_endo = gmsh.model.geo.addPoint(*base_endo_pt)
        base_epi = gmsh.model.geo.addPoint(*base_epi_pt)

        base = gmsh.model.geo.addLine(base_endo, base_epi)
        endo = gmsh.model.geo.addEllipseArc(apex_endo, center, apex_endo, base_endo)
        epi = gmsh.model.geo.addEllipseArc(apex_epi, center, apex_epi, base_epi)

        ll1 = gmsh.model.geo.addCurveLoop([apex, epi, -base, -endo])

        s1 = gmsh.model.geo.addPlaneSurface([ll1])
        out = [(2, s1)]

        rev1 = gmsh.model.geo.revolve([out[0]], center_pt[0], center_pt[1], center_pt[2], 
                            axis_rotation[0], axis_rotation[1], axis_rotation[2], onp.pi/2)
        rev2 = gmsh.model.geo.revolve([out[0]], center_pt[0], center_pt[1], center_pt[2], 
                            axis_rotation[0], axis_rotation[1], axis_rotation[2], -onp.pi/2)
        
    gmsh.model.geo.synchronize()

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    gmsh.model.mesh.generate(3)

    gmsh.write("temp.msh")
    gmsh.finalize()

    mesh = meshio.read("temp.msh")

    epi_idxs = [33, 37, 42, 45]
    endo_idxs = [35, 39, 44, 47]
    base_idxs = [34, 38, 43, 46]
    epi_pts, endo_pts, base_pts = [], [], []
    for i in range(4):
        epi_idx = epi_idxs[i]
        base_idx = base_idxs[i]
        endo_idx = endo_idxs[i]

        epi_pts.append(onp.unique(mesh.cells[epi_idx].data.flatten()))
        base_pts.append(onp.unique(mesh.cells[base_idx].data.flatten()))
        endo_pts.append(onp.unique(mesh.cells[endo_idx].data.flatten()))

    endo_pts = onp.unique(onp.concatenate(endo_pts))
    epi_pts = onp.unique(onp.concatenate(epi_pts))
    base_pts = onp.unique(onp.concatenate(base_pts))
    endo_mask = onp.zeros(mesh.points.shape[0], dtype=bool) 
    epi_mask = onp.zeros(mesh.points.shape[0], dtype=bool)
    base_mask = onp.zeros(mesh.points.shape[0], dtype=bool)
    apex_mask = (mesh.point_data["gmsh:dim_tags"] == [0, apex_epi]).all(axis=1)

    endo_mask[endo_pts] = True
    epi_mask[epi_pts] = True
    base_mask[base_pts] = True

    point_data = {"endo": endo_mask.astype(float),
                  "epi": epi_mask.astype(float),
                  "base": base_mask.astype(float),
                  "apex": apex_mask.astype(float)}

    mesh = gmsh_to_meshio(msh_file, point_data=point_data)
    return mesh
