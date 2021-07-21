Installing DISPATCHES
=====================

Introduction
------------

.. _about-conda:

Using Conda environments
^^^^^^^^^^^^^^^^^^^^^^^^

Conda environments are a way to create and manage multiple sets of packages and/or Python versions on the same system without incurring conflicts.
Each Conda environment is a dedicated directory, separate from other Conda environments and the operating system's own directories, containing its own collection of packages, executables, and Python installation, including the Python interpreter.
Once a Conda environment is *activated*, using the ``conda activate`` command in a terminal/console, the environment's own version of Python will be used to run commands or interactive sessions for the remainder of the session.

For these reasons, Conda environments are especially useful to install and manage multiple projects (and/or multiple *versions* of the same project) on the same computer with minimal effort,
as they provide a way to seamlessly switch between different projects without conflicts.

Using Conda environments is not mandatory to be able to install and use DISPATCHES; however, it is strongly recommended.

To use Conda environments, the ``conda`` package manager is required. Refer to the `Conda installation guide <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ for detailed steps on how to install Conda for your operating system.

General installation
--------------------

If you are going to use DISPATCHES's functionality, but *do not* plan to contribute code to DISPATCHES's codebase directly, choose this option.

#. Create a Conda environment (in this example, named ``dispatches``) where DISPATCHES and its runtime dependencies will be installed:

	.. code-block:: shell

		conda create --name dispatches --yes python=3.8

#. Activate the ``dispatches`` environment:

	.. code-block:: shell

		conda activate dispatches
	
	To verify that the correct environment is active, run the following command:

	.. code-block:: shell

		python -c "import sys; print(sys.executable)"
	
	If the environment was activated correctly, its name should be contained in the path displayed by the above command.

	.. important:: The ``conda activate`` command described above must be run each time a new terminal/console session is started.

#. Install DISPATCHES using ``pip``:

	.. code-block:: shell

		pip install https://github.com/gmlc-dispatches/dispatches/archive/main.zip

#. To verify that the installation was successful, open a Python interpreter and try importing some of DISPATCHES's modules, e.g.:

	.. code-block:: shell

		python
		>>> from dispatches.unit_models import *

For DISPATCHES developers
-------------------------

If you plan to contribute to DISPATCHES's codebase, choose this option.

.. note:: Typically, *contributing to DISPATCHES* will involve opening a Pull Request (PR) in DISPATCHES's repository.

#. Create a Conda environment (in this example, named ``dispatches-dev``) where DISPATCHES and all dependendencies needed for development will be installed, then activate it:

	.. code-block:: shell

		conda create --name dispatches-dev --yes python=3.8 && conda activate dispatches-dev

	.. note:: For more information about using Conda environments, refer to the ":ref:`about-conda`" section above.

#. Clone the DISPATCHES repository to your local development machine using ``git clone``, then enter the newly created ``dispatches`` subdirectory:

	.. code-block:: shell

		git clone https://github.com/gmlc-dispatches/dispatches && cd dispatches

#. Install DISPATCHES and the development dependencies using ``pip`` and the ``requirements-dev.txt`` file:

	.. code-block:: shell

		pip install -r requirements-dev.txt

#. To verify that the installation was successful, try running the DISPATCHES test suite using ``pytest``:

	.. code-block:: shell

		pytest







