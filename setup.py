"""
Project setup with setuptools
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_namespace_packages
import pathlib

# this will come in handy, probably
cwd = pathlib.Path(__file__).parent.resolve()

long_description = """
DISPATCHES is an open-source suite of models for the design and analyis
of tightly-coupled energy systems based on the IDAES-PSE Platform.  The
DISPATCHES project is funded by the U.S. Department of Energy Grid
Modernization Initiative through the Grid Modernization Lab Consortium.
DISPATCHES is developed by researchers at the National Energy Technology
Laboratory, Idaho National Laboratory, Lawrence Berkeley Laboratory,
National Renewable Energy Laboratory, Sandia National Laboratories, and
the University of Notre Dame.
""".replace("\n", " ").strip()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='dispatches',
    url='https://github.com/gmlc-dispatches/dispatches',
    version='0.0.1',
    description='GMLC DISPATCHES software tools',
    long_description=long_description,
    long_description_content_type='text/plain',
    author='DISPATCHES team',
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords="market simulation, chemical engineering, process modeling, hybrid power systems",
    packages=find_namespace_packages(),
    python_requires='>=3.7, <4',
    install_requires=[
        'pytest',  # technically developer, but everyone likes tests
        'idaes-pse',
        'egret @ git+https://github.com/grid-parity-exchange/Egret.git',
        'prescient @ git+https://github.com/grid-parity-exchange/Prescient.git'
    ],
    dependency_links=['git+https://github.com/grid-parity-exchange/Prescient.git#egg=prescient'],
    extras_require={
         'dev': [
             'pytest-cov',
             'Sphinx==3.4.2',
             'sphinx_rtd_theme',
         ],
    },
   package_data={  # Optional
        "": [
            "*.json",
        ],
    },
)
