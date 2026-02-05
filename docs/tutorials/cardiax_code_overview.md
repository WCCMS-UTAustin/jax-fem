
# Overall Structure

Let's take a moment to look at the overall structure of CARDIAX. There are three main components that the user will interact with to solve various finite element problems. There are three main objects to work with:

1. FiniteElement
2. Problem
3. Solver

Each of these are classes that are slowly built upon, using more refined inheritance as the problem becomes more defined. We will go over how each of these objects works as well as some caveats to watch out for.

One should think of finite element problems in a more abstract sense, and this will be reflected by the structure of the code. The `Problem` object handles the PDE, more explicitly, the weak form of the given equation. The `FiniteElement` object represents the disretization of the domain being used to approximate the PDE in `Problem`. Finally, the `Solver` object is responsible for taking the `FiniteElement` and `Problem` objects and actually solving the system of equations that arise from the finite element discretization.

## FiniteElement

The `FiniteElement` object most fundamentally requires the mesh object. This has recently been changed to a pyvista object to allow for all the capabilities from the Visualization Toolkit (VTK) to be used. Along with the mesh object, we give the dimensions of the mesh (for padding reasons), the vector of the FE field (e.g. displacement, temperature, etc.), the element type, and the degree of quadrature to be used. Multiple FE objects can be formed because each one represents a field over the mesh. These "local" attributes are then handled by `Problem`.

## Problem

The `Problem` object is responsible for defining the weak form of the PDE to be solved. This is done through inheritance because you will ALWAYS define a child class from `Problem` where the user specifies the weak form. The `Problem` object then pushes through with that specific object being defined. The boundary conditions are then given as a further specification of the desired PDE. This will become more clear through various examples. If the PDE has multiple fields, the `Problem` object will be able to coordinate these fields together to solve the global system. The main duty is creating the matrix system to be solved through the linearization of the weak form to be passed to the `Solver`.

## Solver

The `Solver` object is responsible for taking the matrix system created by the `Problem` object and actually solving it. The most standard version is the `Newton_Solver` which uses a Newton-Raphson iterative scheme to solve nonlinear problems. Other solvers can be created by inheriting from the `Base_Solver` class and defining the specific solve method. It's built upon Lineax to provide as much information as possible to the user about the iterative scheme for debugging purposes.