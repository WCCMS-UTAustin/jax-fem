
import jax
import jax.numpy as np
import os
import time
import numpy as onp
from pathlib import Path

from cardiax import FiniteElement, Problem, Newton_Solver
from cardiax import rectangle_mesh

class Reaction_Diffusion(Problem):
    # The function 'get_tensor_map' is responsible for the term (sigma*Vm_grad, Q_grad) * dx. 
    # The function 'get_mass_map' is responsible for the term (A_m (C_m (Vm - V_old)/dt + I_ion) - I_s, Q) * dx. 
    # The function 'set_params' makes sure that the Neumann boundary conditions use the most 
    # updated V_old
    def get_tensor_map(self):
        def fn(u_grad, u_old):
            return 1e-5 * u_grad
        return fn
 
    def get_mass_map(self):
        def V_map(u, u_grad, x, u_old):
            return (u - u_old)/self.dt - self.r * u_old * (1 - u_old)

        return V_map

    def set_params(self, params, args):
        self.r, self.dt = args["r"], args["dt"]
    
        sol_u_old = self.fes["u"].convert_dof_to_quad(params[0])
        # forcing_values = self.fes["u"].convert_dof_to_quad(params[1])

        self.internal_vars = {"u": {"sol_u_old": sol_u_old}}
        return

# Save directory
vtk_dir = "./data/"
os.makedirs(vtk_dir, exist_ok=True)

# Params
params = {
    "r": 1.,
    "dt": 1e-1,
}

# Define Monodomain ODE solving
dt = 1e-1
simulation_t = 10.
ts = np.arange(0., simulation_t, dt)

# Specify mesh-related information.
ele_type = 'quad'
Nx, Ny = 40, 40
Lx, Ly = 1.0, 1.0 # domain size
mesh = rectangle_mesh(Nx, Ny, Lx, Ly)

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)

# Specify boundary conditions and problem definitions.
# Dirichlet BC values for thermal problem
def voltage_dirichlet_bottom(point):
    return 0.

# Define monodomain problem
dirichlet_bc_info_u = [[[bottom], [0], [voltage_dirichlet_bottom]]]
fe = FiniteElement(mesh, vec=1, dim=2, ele_type=ele_type, gauss_order=1)
problem = Reaction_Diffusion({"u": fe})

forcing = 1.
pos = np.array([0.5, 0.5])
def gaussian(x):
    return forcing * np.exp(- np.linalg.norm(x[:fe.dim] - pos)/(2 * (1/8)**2))
apply_f = jax.vmap(gaussian)
u_n = apply_f(mesh.points)

solver = Newton_Solver(problem, u_n)

# Start the major loop of time iteration.
toc = time.time()
sols = []
for i in range(len(ts[1:])):
    print(f"\nStep {i + 1}, total step = {len(ts[1:])}")

    # Set parameter and solve for u
    problem.set_params([u_n[:, None]], params)
    u_n, info = solver.solve()
    assert info[0]
    sols.append(u_n)

tic = time.time()
print(f"Time elapsed for {i} iterations: ", tic - toc)

if plotting := True:
    import pyvista as pv
    fig_dir = Path("../../figures/Cardiac_Modeling/Monodomain/")
    os.makedirs(fig_dir, exist_ok=True)

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = sols[0].reshape(-1,)
    warped = mesh.warp_by_scalar("sol", factor=1.)

    pl.add_mesh(warped)
    pl.open_gif(fig_dir / "monodomain_movie.gif")

    for i, s in enumerate(sols):
        pl.clear()
        mesh.point_data["sol"] = s.reshape(-1,)
        warped = mesh.warp_by_scalar("sol", factor=1.)
        pl.add_title(f"t = {ts[i]:.2f}")
        pl.add_mesh(warped, reset_camera=False)
        pl.write_frame()
    pl.close()
