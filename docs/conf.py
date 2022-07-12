##############################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
##############################################################################
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
import logging
from pathlib import Path
import sphinx_rtd_theme
import shutil

from sphinx.application import Sphinx as SphinxApp
# sys.path.insert(0, os.path.abspath('.'))


_logger = logging.getLogger("sphinx.conf")


# -- Project information -----------------------------------------------------

project = 'DISPATCHES'
copyright = '2022, GMLC-DISPATCHES Collaboration'
author = 'GMLC-DISPATCHES Collaboration'

# The full version, including alpha/beta/rc tags
release = '1.1.dev0'
# The short X.Y version
version = '1.1.dev0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#
html_logo = "images/dispatches_logo_only.svg"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#
html_favicon = "_static/favicon.ico"


def _create_notebook_index(
        base_path,
        title: str = "Jupyter notebooks",
        title_underline_char: str = "=",
        indent: str = " " * 3,
    ) -> Path:
    base_path = Path(base_path).resolve()
    title = str(title)
    nb_index_entries = [
        p.relative_to(base_path).with_suffix("")
        for p in sorted(base_path.glob("*.ipynb"))
    ]
    nb_index_path = base_path / "index.rst"

    lines = [
        title,
        title_underline_char * len(title),
        "",
        ".. toctree::",
        f"{indent}:maxdepth: 2",
        "",
    ]

    for entry in nb_index_entries:
        lines.append(f"{indent}{entry}")
    lines.append("")

    text = "\n".join(lines)
    nb_index_path.write_text(text)

    return nb_index_path


def _install_notebooks(app: SphinxApp, search_root_dir=None, dest_subdir: Path = Path("examples")):
    search_root_dir = Path(search_root_dir) if search_root_dir else Path(app.confdir).parent
    dest_dir = (Path(app.srcdir) / dest_subdir).resolve()
    notebook_paths = sorted(
        p for p in search_root_dir.rglob("*.ipynb")
        if (
            not p.parent.name == ".ipynb_checkpoints"
            and dest_dir not in p.parents
        )
    )
    _logger.info(
        f"Found {len(notebook_paths)} .ipynb files in {search_root_dir}"
        f" to be copied to {dest_dir}"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src_path in notebook_paths:
        dst_path = dest_dir / src_path.name
        _logger.debug(f"{src_path} -> {dst_path}")
        if dst_path.is_file():
            dst_path.unlink()
        shutil.copy2(src_path, dst_path)
    
    nb_index_path = _create_notebook_index(dest_dir, title="Examples (Jupyter notebooks)")
    _logger.info(f"Generated index file {nb_index_path}")

    if _logger.isEnabledFor(logging.INFO):
        print(nb_index_path.read_text())


nbsphinx_execute = "never"


def setup(app):
    app.connect("builder-inited", _install_notebooks)
