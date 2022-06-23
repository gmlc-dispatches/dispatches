# DISPATCHES
The **D**esign **I**ntegration and **S**ynthesis **P**latform to **A**dvance **T**ightly **C**oupled **H**ybrid **E**nergy **S**ystems (DISPATCHES),
is developed and used to identify and optimize Integrated Energy Systems for operation within the bulk power system via energy market signals.

DISPATCHES is part of the DOE Grid Modernization Laboratory Consortium (GMLC).

## Project Status
[![Python package](https://github.com/gmlc-dispatches/dispatches/actions/workflows/checks.yml/badge.svg)](https://github.com/gmlc-dispatches/dispatches/actions/workflows/checks.yml)
[![Documentation Status](https://readthedocs.org/projects/dispatches/badge/?version=main)](https://dispatches.readthedocs.io/en/latest/?badge=main)
[![GitHub contributors](https://img.shields.io/github/contributors/gmlc-dispatches/dispatches.svg)](https://github.com/gmlc-dispatches/dispatches/graphs/contributors)
[![Merged PRs](https://img.shields.io/github/issues-pr-closed-raw/gmlc-dispatches/dispatches.svg?label=merged+PRs)](https://github.com/gmlc-dispatches/dispatches/pulls?q=is:pr+is:merged)
[![Issue stats](http://isitmaintained.com/badge/resolution/gmlc-dispatches/dispatches.svg)](http://isitmaintained.com/project/gmlc-dispatches/dispatches)
[![Downloads](https://pepy.tech/badge/dispatches)](https://pepy.tech/project/dispatches)

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

As a developer, to ensure that all the .py files in your workspace have the correct copyright header
info (as defineded in `header_text.txt`), use the `addheader` tool installed by `requirements-dev.txt`
as follows:

```sh
addheader -c .addheader.yml
```

### Documentation

For showing documentation from your code in the Sphinx (.rst) docs, see [the Sphinx autodoc documentation](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc) for details on how to format and give options in your documentation file.

## Funding acknowledgements

This work was conducted as part of the Design Integration and Synthesis Platform to Advance Tightly
Coupled Hybrid Energy Systems (DISPATCHES) project with support through the [Grid Modernization Lab
Consortium](https://www.energy.gov/gmi/grid-modernization-lab-consortium) with funding from the U.S.
Department of Energy’s [Office of Fossil Energy and Carbon Management](https://www.energy.gov/fecm/office-fossil-energy-and-carbon-management),
[Office of Nuclear Energy](https://www.energy.gov/ne/office-nuclear-energy), and [Hydrogen and Fuel Cell Technology Office](https://www.energy.gov/eere/fuelcells/hydrogen-and-fuel-cell-technologies-office).
