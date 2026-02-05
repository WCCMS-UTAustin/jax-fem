
# Poisson Example

## Poisson Equation

Here, we will demonstrate how to solve the Poisson equation in the `cardiax` framework. The Poisson equation is defined as

$$
- \Delta u = f
$$

where we take $u \in H^2(\Omega)$ (for simplicity) and $f \in L^2(\Omega)$. Thus, we have an operator, $\mathcal{L}: H^2(\Omega) \rightarrow L^2(\Omega)$ defined by the laplacian. Now, we define what is called a distrubtion (ie weak form) by integrating after multiplying with a suitable test function. This gives us

$$
\int_\Omega \Delta u v dV = \int_\Omega f v dV
$$

then integrating by parts to move the derivative to the test function, we have

$$
\int_\Omega \nabla u \cdot \nabla v dV - \int_{\partial \Omega} v \nabla u \cdot \mathbf{n} dS = \int_\Omega f v dV
$$

These are the simple equations that we would like to solve within `cardiax`.


## Implementation

### Imports

First, we start with basic imports.

```python
import jax
import jax.numpy as np
import os

from cardiax import rectangle_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
```

### Finite Element discretization

We begin by defining the domain that we want to solve the problem over, $\Omega$. There are a few capabilities in CARDIAX to generate the desired mesh, but this is typically imported from some outside object. For this example, we will solve the problem over a 2D square to allow for visualization of the solution. We will create a unit square with 32 linear quadrilateral elements in each direction. Since they are linear, the gauss_order is set to 1.

```python
# Create the mesh and FE field
Lx, Ly = 1., 1.
mesh = rectangle_mesh(Nx=32, Ny=32, Lx=Lx, Ly=Ly)
fe = FiniteElement(mesh, vec=1, dim=2, ele_type="quad", gauss_order=1)
```

### Define boundary locations.

With the specific PDE defined, we now need to say where the boundaries are located to evaluate the surface integrals and apply the dirichlet BCs. We define the boundaries as lookup functions that return True if the desired point is located on the boundary surface. These are then vectorized to obtain the DoFs on the boundary, so we must use jax.numpy functions.

```python
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
```

With the location functions defined, we combine these to form the BC information. For one dirichlet boundary condition, we give a list of 3 lists. The three lists are [[location_fns], [DoF index], [value_fns]]. Since we are solving for a scalar field, each sublist is of size 1. We then signal that these dirichlet boundary conditions belong with the FE field "u". A similar thing is done with the location functions for the surface integral, but we also say for instance the labelled "bottom" face belongs to location function bottom.

```python
# Combine BC info
bc_left = [[left], [0], [zero_bc]]
bc_right = [[right], [0], [zero_bc]]
dirichlet_bc_info = {"u": [bc_left, bc_right]}
location_fns = {"u": {"bottom": bottom, "top": top}}
```

### Problem Creation

Now that we have the mesh object proprely discretized and boundary facets identified, we can define the problem class that will contain all the information about the Poisson problem. This includes the kernel, source terms, and boundary conditions. We will create a class that inherits from `Problem`. To more easily follow the code, we break up the equation as follows:

$$
\texttt{get\_tensor\_map}(u\_grad) := \int_\Omega (u\_grad) \cdot \nabla v dV \\
\texttt{get\_mass\_map}(u, u\_grad, x) := \int_\Omega f(u, u\_grad, x) v dV \\
\texttt{get\_surface\_maps}(u, u\_grad, x) := \int_{\partial \Omega} v g(u, u\_grad, x) \cdot \mathbf{n} dS
$$

where we carry the same notation as before. Note that $g$ is function, so we have a lot of freedom with setting Neumann and Robin boundary conditions.

```python
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
            val = -np.array([0.*np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)])
            return val
        return mass_map

    # Define potential surface kernels
    # Just sinusoidal here
    def get_surface_maps(self):
        def surface_map1(u, u_grad, x):
            return -np.array([np.sin(5.*x[0])])

        def surface_map2(u, u_grad, x):
            return np.array([np.sin(5.*x[0])])

        return {"u": {"bottom": surface_map1, "top": surface_map2}}
```

The above class is then initiated by feeding in the dictionaries defining the problem: the finite element field discretization, dirichlet boundary info, surface boundary info.

```python
problem = Poisson({"u": fe}, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
```

### Create instance of Newton_Solver

The `problem` constructs the objects needed to solve for the appropriate DoFs. Thus, the `Newton_Solver` requires the problem to initiate itself as well an initial guess that's usually 0s.

```python
solver = Newton_Solver(problem, np.zeros((len(mesh.points), 1)))
```

### Solve the problem

The solver is then called and gives the solution as well as info about the solve. Remember to add the assertion to validate that the solve terminated appropriately. This is added for control flow loops that will be demonstrated later where you may have an ill-defined problem.

```python
sol, info = solver.solve()
assert info[0]
```

![alt text](../../../tutorials/Introductory/poisson/poisson_initial.png)

### JIT Speed

The beauty of these solvers being based in JAX is the ability to reuse JIT compiled functions. By slightly modifying the above problem class, we can solve many instances of problems very quickly. If we instead replace the `get_surface_maps` function above by this one, we now have the added value of `a` which describes the magnitude of the Neumann bc.

```python
# Define potential surface kernels
# Just sinusoidal here
def get_surface_maps(self):
    def surface_map1(u, u_grad, x, a):
        return -np.sin(a*x[0]).reshape(1,)

    def surface_map2(u, u_grad, x, a):
        return np.sin(a*x[0]).reshape(1,)

    return {"u": {"bottom": surface_map1, "top": surface_map2}}
```

The first call of the solve will be slowest since it JIT compiles the functions. After the first call, the subsequent solves should be much faster. Thus, we chain the solves like

```python
sols = []
tic = time.time()
for n in range(num_frames):
    problem.set_internal_vars_surfaces({"u": {"bottom": {"a": np.array([a_values[n]])}, "top": {"a": np.array([a_values[n]])}}})
    sol, info = solver.solve()
    assert info[0]
    sols.append(onp.array(sol))
toc = time.time()
```

To solve the first problem takes 2.97 seconds. Once JIT compiled, we solved 100 subsequent problems in 3.57 seconds, averaging 0.0357 per solve. The key is to keep arguments functional, so we minimally have to JIT compile these functions.

![alt text](../../../tutorials/Introductory/poisson/poisson_movie.gif)
