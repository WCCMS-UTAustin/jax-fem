
# NeoHookean

## NeoHookean Equation

After looking at a simple [linear problem](poisson.md), we move towards a nonlinear problem. This is where the power of the GPU based computations really shine. We are solving a conservation of linear momentum equation

$$
\rho_0 \ddot{\varphi} = \nabla \cdot \mathbf{P} + \rho_0 \mathbf{b}
$$

To unpack the notation, $\varphi$ represents the motion of the body. Dynamics will be gone over in [dynamics](../Advanced/dynamic_problems.md), so we will assume quasi-static. Thus, $\varphi$ is independent of time for further simplification, assume no body force $\mathbf{b} = 0$ to get

$$
0 = \nabla \cdot \mathbf{P}
$$

We can obtain a displacement field $\mathbf{u} = \varphi(\mathbf{X}) - \mathbf{X}$ with $\mathbf{X}$ as the reference coordinate, where $\mathbf{u}$ will be the variable of interest. Then we have the deformation gradient $\mathbf{F}(\mathbf{X}) = \nabla \mathbf{\varphi} = \nabla \mathbf{u} + \mathbf{I}$. To obtain the First Piola Kirchhoff stress tensor, we need to define the strain energy which is where this becomes hyperelasticity. We define the strain energy as

$$
\Psi(\mathbf{F}) = C(\mathbf{F}^T \mathbf{F} - 3 - 2\ln(J)) + D (J - 1)^2
$$

where $J = \det(\mathbf{F})$. Now differentiating with respect to $\mathbf{F}$ will give us

$$
\mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}
$$

The major convience is that this derivative can be computed via `jax.grad`, so little derivation is required.

## Implementation

Now we will go over how to solve this nonlinear problem.

### Imports

First, we start with basic imports.

```python
import jax
import jax.numpy as np
import os

from cardiax import box_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
```

### Finite Element Discretization

For the finite element discretization, we will create a beam with our box mesh.

```python
# Specify mesh-related information (first-order hexahedron element).
mesh = box_mesh(Nx=10, Ny=10, Nz=50, Lx=1., Ly=1., Lz=5.)
fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = "hexahedron", gauss_order = 1)
```

### Boundary conditions

```python
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
```

### Problem Construction

```python
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

problem = HyperElasticity({"u": fe},
                          dirichlet_bc_info=dirichlet_bc_info,
                          location_fns=location_fns)

```

### Solver

```python
solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))

forces = np.linspace(0, .025, 21, endpoint=True)

sols = []
for f in forces:
    problem.set_internal_vars_surfaces({"u": {"top": {"t": np.array([f])}}})
    sol, info = solver.solve(max_iter=50)
    assert info[0]
    sols.append(sol)
```

