# Welcome to CARDIAX

## What is CARDIAX?

We are a finite element package foundationally built on [JAX](https://github.com/jax-ml/jax), forked from [JAX-FEM](https://github.com/deepmodeling/jax-fem). The code utilizes GPUs to accelerate standard finite element code. The code is being developed and maintained by the [Willerson Center for Cardiovascular Modeling and Simulation](https://wccms.oden.utexas.edu/). We are focused on solving problems relevant for cardiac mechanics and the host of PDEs that come along with it. This limits the scope of new updates and features requests towards these targeted applications.

# Getting Started

Depending on your desired level of involvement in getting into the details of coding. There are two options when using CARDIAX. There is the standard coding examples which can be found in the tutorial listed below, but there is also a template yaml file that is used as a controller at a broader level. If there are PDEs that you would like to be added to the list, please create an issue to see if it's possible (look at limitations before asking though).


# Installation

Make sure you download JAX appropriately, found [here](https://docs.jax.dev/en/latest/installation.html). After that JAX installation is working, install CARDIAX with pip. For now, clone the repo and use

`
pip install . -e
`

This will make CARDIAX be installed as an interactive environment, so as you update the library, the pip install stays up to date. Soon will be updated for a true pip install.

# API

To see the functionality of CARDIAX, the API can be found [here]().

# Tutorials

To begin with using CARDIAX, we recommend going through some of the tutorials listed here. These will have coding and template file examples to guide you through the levels of complexity with the problems at hand. We recommend all users to at least quickly glance at tutorials even if familiar to become familiar with the style of the code.

These tutorials are walkthroughs on how to use and think about the code. The exact code can be found in the CARDIAX/tutorials directory. There are also demos available for more specific problems, but with less of a walkthrough flavor.

# Limitations

Since CARDIAX is being maintained by PhD students focused on developing new computational methods for specific cardiovascular applications, the scope of the code is not intended to be all-encompassing. It is a foundational tool that will be built upon to further the research of (WCCMS)[https://wccms.oden.utexas.edu/]. Here are some of the most important limitations to consider:

1. Memory Capacity
2. Multiple Finite Element Fields
3. Explicit Matrix Formation

For a more in-depth explanation on these limitations, see [here](limitations.md).