
from pathlib import Path
import numpy as onp
import jax
import os
import jax.numpy as np

from cardiax import FiniteElement, Newton_Solver, Problem
from cardiax import box_mesh, get_F

class Myo(Problem):

    def get_tensor_map(self):
        def psi(F, f, s, n):
            f = f[:, None]
            s = s[:, None]
            n = n[:, None]
            J = np.linalg.det(F)
            C = F.T @ F * np.cbrt(J)**(-2)
            E_tilde = 1/2 * (C - np.eye(self.dim['u']))
            E11 = f.T @ E_tilde @ f
            E12 = f.T @ E_tilde @ s
            E13 = f.T @ E_tilde @ n
            E22 = s.T @ E_tilde @ s
            E23 = s.T @ E_tilde @ n
            E33 = n.T @ E_tilde @ n

            Q = self.A1 * E11**2 \
            + self.A2 * (E22**2 + E33**2 + 2*E23**2) \
            + self.A3 * (E12**2 + E13**2)
            
            psi_dev = self.c/2 * (np.exp(self.alpha * Q[0, 0]) - 1)
            psi_vol = self.K/2 * ( (J**2 - 1)/2 - np.log(J))
            return psi_dev + psi_vol
        
        P_fn = jax.grad(psi)

        def S_act(F, f, TCa):
            f = f[:, None]
            lamb = np.sqrt(f.T @ F.T @ F @ f)
            S = TCa * 1e3* (1 + self.beta * (lamb - 1))/(lamb ** 2) * f @ f.T
            return S

        def first_PK_stress(u_grad, f, s, n, TCa):
            F = u_grad + np.eye(self.dim["u"])
            P_psi = P_fn(F, f, s, n)
            P_act = F @ S_act(F, f, TCa)
            return P_psi + P_act

        return first_PK_stress

    def get_surface_maps(self):

        def pressure(u, u_grad, x, normal, p):
            F = u_grad + np.eye(len(x))
            J = np.linalg.det(F)
            F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                    [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                    [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]]).T
            val = p * J * F_inv.T @ normal
            return val
    
        return {"u": {"top": pressure}}

    def set_params(self):
        self.c = 1522.083
        self.A1 = 12.
        self.A2 = 8.
        self.A3 = 26.
        self.K = 1e5
        self.alpha = 2.125
        self.beta = 1.4
        return

mesh = box_mesh()
fe = FiniteElement(mesh, vec=3, dim=3, ele_type="hexahedron", gauss_order=1)

# Adding fiber directions to Problem
def_fiber = lambda x: np.array([np.cos(x), 
                                np.sin(x), 
                                0])

def_sheet = lambda x: np.array([-np.sin(x), 
                                np.cos(x), 
                                0])

def_normal = lambda x: np.array([0., 0., 1.])

def theta_val(x):
    return (60 - (1 - x[2]) * 120) * np.pi / 180

cell_centers = mesh.cell_centers()
pts = cell_centers.points
thetas = jax.vmap(theta_val)(pts)

fibers = jax.vmap(def_fiber)(thetas)
sheets = jax.vmap(def_sheet)(thetas)
normals = jax.vmap(def_normal)(thetas)

def bottom(x):
    return np.isclose(x[2], 0., 1e-5)

def top(x):
    return np.isclose(x[2], 1., 1e-5)

def zero_val(x):
    return 0.

bc_bottom = [[bottom]*3, [0, 1, 2], [zero_val]*3]

problem = Myo({"u": fe}, dirichlet_bc_info={"u": [bc_bottom]}, location_fns={"u": {"top": top}})
problem.set_params()
problem.set_internal_vars({"u": {"fibers": fibers, "sheets": sheets, "normals": normals, "TCa": np.array([0.])}})

top_normals = fe.get_surface_normals(top)
surf_vars = {"u": {"top": {"normals": top_normals, "p": np.array([0.])}}}
problem.set_internal_vars_surfaces(surf_vars)

solver = Newton_Solver(problem, initial_guess=onp.zeros((problem.num_total_dofs_all_vars)))

sols = []
ps = np.linspace(0., 5e3, 51, endpoint=True)
for p in ps:
    problem.set_internal_vars_surfaces({"u": {"top": {"normals": top_normals, "p": np.array([p])}}})
    sol, info = solver.solve()
    print("Pressure: ", p)
    assert info[0]
    sols.append(onp.array(sol))

TCas = np.linspace(0., 10., 101, endpoint=True)
for TCa in TCas:
    problem.set_internal_vars({"u": {"fibers": fibers, "sheets": sheets, "normals": normals, "TCa": np.array([TCa])}})
    solver.initial_guess = sol
    sol, info = solver.solve()
    print("TCa: ", TCa)
    assert info[0]
    sols.append(onp.array(sol))

# To save
if plotting := True:
    ps1 = np.hstack((ps, ps[-1] * np.ones_like(TCas)))
    TCas1 = np.hstack((np.zeros_like(ps), TCas))
    vec = np.vstack((ps1, TCas1)).T

    import os
    import pyvista as pv
    fig_dir = Path("../../figures/Cardiac_Modeling/Myocardium/")
    os.makedirs(fig_dir, exist_ok=True)
    mesh.cell_data["fibers"] = onp.array(fibers).reshape((-1, 3))

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = onp.array(sols[0]).reshape(-1, fe.dim)
    warped = mesh.warp_by_vector("sol", factor=1.)
    glyph = warped.glyph(orient="fibers", scale=False, factor=0.1)

    pl.add_mesh(warped)
    pl.add_mesh(glyph, color="red")
    pl.open_gif(fig_dir / "myo_movie.gif")

    for i, s in enumerate(sols):
        pl.clear()
        Fs = get_F(fe, s)
        Fs = Fs.mean(axis=1)
        fibers_new = onp.einsum("ijk, ij-> ik", Fs, fibers)
        mesh.cell_data["fibers"] = onp.array(fibers_new).reshape((-1, 3))
        mesh.point_data["sol"] = onp.array(s).reshape(-1, fe.dim)
        warped = mesh.warp_by_vector("sol", factor=1.)
        glyph = warped.glyph(orient="fibers", scale=False, factor=0.1)

        pl.add_title(f"Pressure, TCas = {vec[i, 0]}, {vec[i, 1]:.1f}")
        pl.add_mesh(warped, reset_camera=False, opacity=0.5)
        pl.add_mesh(glyph, color="red", reset_camera=False)
        pl.write_frame()
    pl.close()
