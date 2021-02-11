"""
Project setup with setuptools
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_namespace_packages
import pathlib

cwd = pathlib.Path(__file__).parent.resolve()  # this will come in handy, probably

long_description = """ProteusLib is an open-source, integrated suite of predictive multi-scale models
for design and optimization of water treatment processes and systems. Specifically, ProteusLib is a new
library of water treatment-specific property, process unit, and network models that depend on the IDAES Platform,
an open source, next generation process systems engineering platform developed at the National Energy Technology
Laboratory with other partners. The ProteusLib project is funded by the NAWI  as a part of U.S. Department of
Energy’s Energy-Water Desalination Hub. The goal of ProteusLib is to assist the hub and the broader water R&D
community in assessing existing and emerging water treatment technologies by 1) providing predictive capabilities
involving the design, optimization, and performance of water treatment systems that will lead to improved energy
efficiency and lower cost, 2) advancing the state of the art for the design of water treatment components, systems
and networks to be comparable with, or even surpass, that in the chemical industry, and 3) disseminating these tools
for active use by water treatment researchers and engineers.""".replace("\n", " ").strip()

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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords="market simulation, chemical engineering, process modeling, hybrid power systems",
    packages=find_namespace_packages(),
    python_requires='>=3.6, <4',
    install_requires=[
        'pytest',  # technically developer, but everyone likes tests
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
