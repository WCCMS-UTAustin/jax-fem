# Import some useful modules.
import jax
import jax.numpy as np
import os
from pathlib import Path

from cardiax import box_mesh
from cardiax import Problem
from cardiax import Newton_Solver
from cardiax import FiniteElement

class HyperElasticity(Problem):

    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (I1 - 3. - 2 * np.log(J)) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(u_grad.shape[0])
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
    def get_surface_maps(self):

        def surface_map(u, u_grad, x, t):
            return np.array([t[0], 0., 0.])
                
        return {"u": {"top": surface_map}}

# Specify mesh-related information (first-order hexahedron element).
mesh = box_mesh(Nx=10, Ny=10, Nz=50, Lx=1., Ly=1., Lz=5.)
fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = "hexahedron", gauss_order = 1)

# Define boundary locations.
def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def top(point):
    return np.isclose(point[2], 5., atol=1e-5)

# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.

bc1 = [[bottom] * 3, [0, 1, 2],
        [zero_dirichlet_val] * 3]
dirichlet_bc_info = {"u": [bc1]}
location_fns = {"u": {"top": top}}

problem = HyperElasticity({"u": fe},
                          dirichlet_bc_info=dirichlet_bc_info,
                          location_fns=location_fns)

solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))

forces = np.linspace(0, .025, 21, endpoint=True)

problem.set_internal_vars_surfaces({"u": {"top": {"t": np.array([forces[-1]])}}})
sol0 = solver.solve(max_iter=50)[0]

sols = []
for f in forces:
    problem.set_internal_vars_surfaces({"u": {"top": {"t": np.array([f])}}})
    sol, info = solver.solve(max_iter=50)
    assert info[0]
    sols.append(sol)

if plotting := True:
    fig_dir = Path("../../figures/Introductory/neohookean/")
    import pyvista as pv
    import numpy as onp

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = onp.array(sol0).reshape(-1, fe.dim)
    warped = mesh.warp_by_vector("sol", factor=1.)
    pl.add_mesh(warped)
    pl.camera_position = 'xy'
    pl.camera.roll += 90
    pl.camera.azimuth += 30
    pl.camera.elevation += 30.
    pl.screenshot(fig_dir / "beam_disp.png")
    pl.close()

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = onp.array(sols[0]).reshape(-1, fe.dim)
    warped = mesh.warp_by_vector("sol", factor=1.)

    pl.add_mesh(warped)
    pl.camera_position = 'xy'
    pl.camera.roll += 90
    pl.camera.azimuth += 30
    pl.camera.elevation += 30.
    pl.open_gif(fig_dir / "beam_movie.gif")

    for i, s in enumerate(sols):
        pl.clear()
        mesh.point_data["sol"] = onp.array(s).reshape(-1, fe.dim)
        warped = mesh.warp_by_vector("sol", factor=1.)
        pl.add_title(f"force = {forces[i]:.3f}")
        pl.add_mesh(warped, reset_camera=False)
        pl.write_frame()
    pl.close()
