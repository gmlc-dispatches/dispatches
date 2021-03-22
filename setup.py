"""
Project setup with setuptools
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_namespace_packages
import pathlib
import re

# this will come in handy, probably
cwd = pathlib.Path(__file__).parent.resolve()

# Parse long description from README.md file
with open("README.md") as f:
    lines, capture = [], False
    for line in f:
        s = line.strip()
        if re.match(r"#\s*[Aa]bout", s):
            capture = True
        elif re.match("^#", s):
            break
        elif capture is True:
            lines.append(s)
    if lines:
        long_description = " ".join(lines)
    else:
        long_description = "DISPATCHES project"


def read_requirements(input_file):
    """Build list of requirements from a requirements.txt file
    """
    req = []
    for line in input_file:
        s = line.strip()
        c = s.find("#")  # look for comment
        if c != 0:  # no comment (-1) or comment after start (> 0)
            if c > 0:  # strip trailing comment
                s = s[:c]
            req.append(s)
    return req


with open("requirements.txt") as f:
    package_list = read_requirements(f)

with open("requirements-dev.txt") as f:
    dev_package_list = read_requirements(f)

########################################################################################

setup(
    name="dispatches",
    url="https://github.com/gmlc-dispatches/dispatches",
    version="0.0.1",
    description="GMLC DISPATCHES software tools",
    long_description=long_description,
    long_description_content_type="text/plain",
    author="DISPATCHES team",
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
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
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="market simulation, chemical engineering, process modeling, hybrid power systems",
    packages=find_namespace_packages(),
    python_requires=">=3.6, <4",
    install_requires=package_list,
    dependency_links=[
        "git+https://github.com/grid-parity-exchange/Prescient.git#egg=gridx-prescient"
    ],
    package_data={"": ["*.json"]},  # Optional
    extras_require={"dev": dev_package_list},
)
