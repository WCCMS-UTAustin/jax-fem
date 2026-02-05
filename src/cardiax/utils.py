import meshio
import numpy as onp

from cardiax import FiniteElement

# This function can be simplified since we are using meshio.Mesh objects now
# Also need to be careful and check the file type because I've had experience with
# different vector readings depending on the .* type
def save_sol(fe: FiniteElement, sol, sol_file, cell_type, cell_infos=None, point_infos=None):
    """ saves solution to visualize in paraview

    Saves the solution at lagrange points and outputs to a .vtu file that is readable with Paraview.
    I'm running into issues with support for HEX64 elements in VTK/VTU files, so cell connectivity
    is going to be reduced to a 'linearized' representation of the mesh.

    Parameters
    ----------
    fe : cardiax.fe.FiniteElement
        _description_
    sol : onp.ndarray
        output from solver(); dofs for a given problem.
    sol_file : _type_
        _description_
    cell_infos : _type_, optional
        _description_, by default None
    point_infos : _type_, optional
        _description_, by default None
    """
    
    points = fe.mesh.points
    cells = fe.mesh.cells

    # create mesh object, with solution saved at points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
    out_mesh.point_data['sol'] = onp.array(sol, dtype=onp.float64) # was 32bit precision
    
    # NOTE: I'm not 100% sure if cell_infos or point_infos need to be modified depending on the type of simulation.
    #       the code below is remaining untouched for now, please update if it throws an errors
    if cell_infos is not None:
        for cell_info in cell_infos:
            # name, data = cell_info
            for name, data in cell_info.items():
            # TODO: vector-valued cell data
                # assert data.shape == (fe.num_cells,), f"cell data wrong shape, get {data.shape}, while num_cells = {fe.num_cells}"
                out_mesh.cell_data[name] = [onp.array(data, dtype=onp.float32)]
    if point_infos is not None:
        for point_info in point_infos:
            # name, data = point_info
            for name, data in point_info.items():
                data = data.reshape(-1, fe.vec)
                print("d",len(data))
                print("s",len(sol))
                assert len(data) == len(sol), "point data wrong shape!"
                out_mesh.point_data[name] = onp.array(data, dtype=onp.float32)
    out_mesh.write(sol_file)

