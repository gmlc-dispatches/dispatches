# About
This is the primary repository for distributing dispatches software tools.

## Build Status

[![Python package](https://github.com/gmlc-dispatches/dispatches/actions/workflows/checks.yml/badge.svg)](https://github.com/gmlc-dispatches/dispatches/actions/workflows/checks.yml)
[![Documentation Status](https://readthedocs.org/projects/dispatches/badge/?version=main)](https://dispatches.readthedocs.io/en/latest/?badge=main)

## Description

DISPATCHES, the Design Integration and Synthesis Platform to Advance Tightly Coupled Hybrid Energy Systems,
will be developed and used to identify and optimize Integrated Energy Systems for operation within the bulk 
power system via energy market signals.

DISPATCHES is part of the DOE Grid Modernization Laboratory Consortium (GMLC).

## Getting started

### Using Conda environments

The recommended way to install DISPATCHES is to use a Conda environment.

A Conda environment is a separate installation directory where packages and even different Python versions can be installed
without conflicting with other Python versions installed on the system, or other environments.

To create a Conda environment, the `conda` command should be installed and configured for your operating system.
Detailed steps to install and configure `conda` are available [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### For developers

(Recommended) Create a dedicated Conda environment for development work:

```sh
conda create -n dispatches-dev python=3.8 pip --yes
conda activate dispatches-dev
```

Clone the repository and enter the `dispatches` directory:

```sh
git clone https://github.com/gmlc-dispatches/dispatches
cd dispatches
```

Install the Python package and all dependencies required for development work using pip and the `requirements-dev.txt` file:

```sh
pip install -r requirements-dev.txt
```

The developer installation will install the cloned directory in editable mode (as opposed to the default behavior of installing a copy of it),
which means that any modification made to the code in the cloned directory
(including switching to a different branch with `git switch`/`git checkout`, or updating the repository with the latest changes using `git pull`) will be available when using the package in Python,
regardless of e.g. the current working directory.

To test that the installation was successful, run the test suite using the `pytest` command:

```sh
pytest
```

### Documentation

For showing documentation from your code in the Sphinx (.rst) docs, see [the Sphinx autodoc documentation](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc) for details on how to format and give options in your documentation file.
