# Import some generally useful packages.
import jax.numpy as np
import numpy as onp
import time

from cardiax import rectangle_mesh
from cardiax import FiniteElement, Problem, Newton_Solver

# Define the Poisson problem
class Poisson(Problem):

    # This defines the kernel
    # \int \nabla u \cdot \nabla v dx
    # the "\cdot \nabla v" is fixed, so only provide the \nabla u
    def get_tensor_map(self):
        return lambda u_grad: u_grad
    
    # Define the source term f
    # For the Poisson problem, using gaussian here
    def get_mass_map(self):
        def mass_map(u, u_grad, x):
            val = -np.array([15.*np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)])
            return val
        return mass_map

    # Define potential surface kernels
    # Just sinusoidal here
    def get_surface_maps(self):
        def surface_map1(u, u_grad, x, a):
            return -np.sin(a*x[0]).reshape(1,)

        def surface_map2(u, u_grad, x, a):
            return np.sin(a*x[0]).reshape(1,)

        return {"u": {"bottom": surface_map1, "top": surface_map2}}
    
# Create the mesh and FE field
Lx, Ly = 1., 1.
mesh = rectangle_mesh(Nx=32, Ny=32, Lx=Lx, Ly=Ly)
fe = FiniteElement(mesh, vec=1, dim=2, ele_type="quad", gauss_order=1)

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)

def top(point):
    return np.isclose(point[1], Ly, atol=1e-5)

# Define boundary values to assign (homogeneous)
def zero_bc(point):
    return 0.

# Combine BC info
bc_left = [[left], [0], [zero_bc]]
bc_right = [[right], [0], [zero_bc]]
dirichlet_bc_info = {"u": [bc_left, bc_right]}
location_fns = {"u": {"bottom": bottom, "top": top}}

problem = Poisson({"u": fe}, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
problem.set_internal_vars_surfaces({"u": {"bottom": {"a": np.array([5.])}, "top": {"a": np.array([5.])}}})

# Create instance of Newton_Solver
solver = Newton_Solver(problem, np.zeros((len(mesh.points), 1)))

# Solve the problem
# sol, info = solver.solve(atol=1e-6)
# assert info[0]

### Solving for multiple frames
# Start from a = -5, go to 5 and back to -5
num_frames = 100
a_values = np.linspace(-5., 5., num_frames//2, endpoint=True)
a_values = np.concatenate((a_values, a_values[::-1]))

n = -1
problem.set_internal_vars_surfaces({"u": {"bottom": {"a": np.array([a_values[n]])}, "top": {"a": np.array([a_values[n]])}}})
tic = time.time()
sol0, info = solver.solve()
assert info[0]
toc0 = time.time() - tic

sols = []
tic = time.time()
for n in range(num_frames):
    problem.set_internal_vars_surfaces({"u": {"bottom": {"a": np.array([a_values[n]])}, "top": {"a": np.array([a_values[n]])}}})
    sol, info = solver.solve()
    assert info[0]
    sols.append(onp.array(sol))
toc = time.time()

print("Initial solve time:", toc0)
print("Total solve time for all frames:", toc - tic)
print("Average time per frame:", (toc - tic)/num_frames)

if plotting := False:
    import pyvista as pv

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = sol0.reshape(-1,)
    warped = mesh.warp_by_scalar("sol", factor=1.)
    pl.add_mesh(warped)
    pl.screenshot("poisson_initial.png")
    pl.close()

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = sols[0].reshape(-1,)
    warped = mesh.warp_by_scalar("sol", factor=1.)

    pl.add_mesh(warped)
    pl.open_gif("poisson_movie.gif")

    for i, s in enumerate(sols):
        pl.clear()
        mesh.point_data["sol"] = s.reshape(-1,)
        warped = mesh.warp_by_scalar("sol", factor=1.)
        pl.add_title(f"a = {a_values[i]:.2f}")
        pl.add_mesh(warped, reset_camera=False, cmap="reds")
        pl.write_frame()
    pl.close()

