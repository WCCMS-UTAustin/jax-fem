
import jax
import jax.numpy as np
import numpy as onp
from pathlib import Path

from cardiax import prolate_spheroid_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
from cardiax import get_grad

alpha_endo = 60.
alpha_epi = -60.
beta_endo = 0.
beta_epi = 0.

def rotation_around_vec(axis, angle_radians):
    """
    Rotates a 3D vector around a given axis by a specified angle using Rodrigues' rotation formula.

    Args:
        vector (np.ndarray): The 3D vector to be rotated.
        axis (np.ndarray): The 3D vector representing the axis of rotation.
        angle_radians (float): The rotation angle in radians.

    Returns:
        np.ndarray: The rotated 3D vector.
    """
    
    W = np.array([[0., -axis[2], axis[1]],
                  [axis[2], 0., -axis[0]],
                  [-axis[1], axis[0], 0.]])
    R = (np.eye(3) + np.sin(angle_radians) * W +
         (1 - np.cos(angle_radians)) * np.matmul(W, W))
    return R

class Laplace(Problem):

    def get_tensor_map(self):
        return lambda x: x
    
mesh = prolate_spheroid_mesh("PS.vtk", cl_min=0.1, cl_max=0.2)

fe = FiniteElement(mesh, vec=1, dim=3, ele_type="tetra", gauss_order=1)

base_pts = mesh.points[mesh.point_data["base"] == 1]
apex_pts = mesh.points[mesh.point_data["apex"] == 1]
endo_pts = mesh.points[mesh.point_data["endo"] == 1]
epi_pts = mesh.points[mesh.point_data["epi"] == 1]

def base_loc(point):
    return np.isclose(point, base_pts, atol=1e-6).all(axis=1).any()

def apex_loc(point):
    return np.isclose(point, apex_pts, atol=1e-6).all(axis=1).any()

def endo_loc(point):
    return np.isclose(point, endo_pts, atol=1e-6).all(axis=1).any()

def epi_loc(point):
    return np.isclose(point, epi_pts, atol=1e-6).all(axis=1).any()

def ones(point):
    return 1.

def zeros(point):
    return 0.

bc_phi1 = [[epi_loc], [0], [ones]]
bc_phi2 = [[endo_loc], [0], [zeros]]
bcs_phi = {"phi": [bc_phi1, bc_phi2]}

bc_psi1 = [[base_loc], [0], [ones]]
bc_psi2 = [[apex_loc], [0], [zeros]]
bcs_psi = {"psi": [bc_psi1, bc_psi2]}

problem0 = Laplace({"phi": fe}, dirichlet_bc_info=bcs_phi)
solver0 = Newton_Solver(problem0, initial_guess=onp.zeros((problem0.num_total_dofs_all_vars)))

problem1 = Laplace({"psi": fe}, dirichlet_bc_info=bcs_psi)
solver1 = Newton_Solver(problem1, initial_guess=onp.zeros((problem1.num_total_dofs_all_vars)))

phi, info = solver0.solve()
psi, info = solver1.solve()

mesh.point_data["psi"] = onp.array(psi).reshape(-1)
mesh.point_data["phi"] = onp.array(phi).reshape(-1)

et_vecs = get_grad(fe, phi).mean(axis=1)[:, 0, :]
et_vecs = et_vecs / np.linalg.norm(et_vecs, axis=-1, keepdims=True)

k_vecs = get_grad(fe, psi).mean(axis=1)[:, 0, :]
k_vecs = k_vecs / np.linalg.norm(k_vecs, axis=-1, keepdims=True)

dot_prod = np.einsum("ab, ab->a", et_vecs, k_vecs)
en_vecs = k_vecs - dot_prod[:, None] * et_vecs
en_vecs = en_vecs / np.linalg.norm(en_vecs, axis=-1, keepdims=True)

el_vecs = np.cross(-et_vecs, en_vecs)

mesh.cell_data["fibers"] = onp.array(el_vecs)
mesh.cell_data["sheets"] = onp.array(et_vecs)
mesh.cell_data["normals"] = onp.array(en_vecs)

phi_dof = fe.convert_dof_to_quad(phi[:, None]).mean(axis=1)
alpha_new = alpha_endo * (1 - phi_dof) + alpha_epi * phi_dof
beta_new = beta_endo * (1 - phi_dof) + beta_epi * phi_dof

e_vecs = np.stack([el_vecs, et_vecs, en_vecs], axis=-1)
Rs = jax.vmap(rotation_around_vec)(et_vecs, np.radians(alpha_new))

vecs = jax.vmap(lambda R, e: R @ e)(Rs, e_vecs)

mesh.cell_data["fibers_rotated"] = onp.array(vecs[:, :, 0])
mesh.cell_data["sheets_rotated"] = onp.array(vecs[:, :, 1])
mesh.cell_data["normals_rotated"] = onp.array(vecs[:, :, 2])

if plotting := True:
    import os
    import pyvista as pv
    fig_dir = Path("../../figures/Cardiac_Modeling/LDRBM/")
    os.makedirs(fig_dir, exist_ok=True)

    boundaries = onp.zeros_like((mesh.point_data["base"]))
    boundaries[mesh.point_data["base"] == 1] = 1
    boundaries[mesh.point_data["apex"] == 1] = 2
    boundaries[mesh.point_data["endo"] == 1] = 3
    boundaries[mesh.point_data["epi"] == 1] = 4
    mesh.point_data["boundaries"] = boundaries

    pl = pv.Plotter(off_screen=True)
    custom_cmap = pv.LookupTable(scalar_range=(1, 4))
    custom_cmap.cmap = ["red", "black", "blue", "green"]
    pl.add_mesh(mesh, scalars="boundaries", cmap=custom_cmap, 
                show_scalar_bar=True, opacity=.5)
    pl.screenshot(fig_dir / "PS_mesh.png")    
    pl.close()

    pl = pv.Plotter(shape=(1, 2), off_screen=True)
    pl.subplot(0, 0)
    pl.add_mesh(mesh, scalars="phi", cmap="viridis", 
                show_scalar_bar=True, copy_mesh=True)
    pl.add_title("Phi: Endo to Epi")
    pl.subplot(0, 1)
    pl.add_mesh(mesh, scalars="psi", cmap="viridis", 
                show_scalar_bar=True, copy_mesh=True)
    pl.add_title("Psi: Base to Apex")
    pl.screenshot(fig_dir / "Lap_PS.png")
    pl.close()

    fibers = mesh.glyph(orient="fibers", scale=False, factor=1., tolerance=0.05)
    sheets = mesh.glyph(orient="sheets", scale=False, factor=1., tolerance=0.05)
    normals = mesh.glyph(orient="normals", scale=False, factor=1., tolerance=0.05)
    pl = pv.Plotter(shape=(1, 3), off_screen=True)
    pl.subplot(0, 0)
    pl.add_mesh(mesh, scalars=None, opacity=0.5)
    pl.add_mesh(fibers, color="red")
    pl.add_title("Fibers")
    pl.camera.zoom(1.5)
    pl.subplot(0, 1)
    pl.add_mesh(mesh, scalars=None, opacity=0.5)
    pl.add_mesh(sheets, color="blue")
    pl.add_title("Sheets")
    pl.camera.zoom(1.5)
    pl.subplot(0, 2)
    pl.add_mesh(mesh, scalars=None, opacity=0.5)
    pl.add_mesh(normals, color="green")
    pl.add_title("Normals")
    pl.camera.zoom(1.5)
    pl.screenshot(fig_dir / "Fibers_LDRBM.png")    
    pl.close()

    fibers_rotated = mesh.glyph(orient="fibers_rotated", scale=False, factor=1., tolerance=0.05)
    sheets_rotated = mesh.glyph(orient="sheets_rotated", scale=False, factor=1., tolerance=0.05)
    normals_rotated = mesh.glyph(orient="normals_rotated", scale=False, factor=1., tolerance=0.05)
    pl = pv.Plotter(shape=(1, 3), off_screen=True)
    pl.subplot(0, 0)
    pl.add_mesh(mesh, scalars=None, opacity=0.5)
    pl.add_mesh(fibers_rotated, color="red")
    pl.add_title("Rotated Fibers")
    pl.camera.zoom(1.5)
    pl.subplot(0, 1)
    pl.add_mesh(mesh, scalars=None, opacity=0.5)
    pl.add_mesh(sheets_rotated, color="blue")
    pl.add_title("Rotated Sheets")
    pl.camera.zoom(1.5)
    pl.subplot(0, 2)
    pl.add_mesh(mesh, scalars=None, opacity=0.5)
    pl.add_mesh(normals_rotated, color="green")
    pl.add_title("Rotated Normals")
    pl.camera.zoom(1.5)
    pl.screenshot(fig_dir / "Fibers_Rotated_LDRBM.png")    
    pl.close()

