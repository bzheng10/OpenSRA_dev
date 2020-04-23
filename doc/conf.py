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
import os, sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname(os.path.abspath('.'))+r'/src/')


# -- Project information -----------------------------------------------------

project = 'OpenSRA'
copyright = '2020, Slate Geotechnical Consultants and The Regents of the University of California'
author = 'Barry Zheng'

# The full version, including alpha/beta/rc tags
release = '(20XX)'


# -- Keywords -----------------------------------------------------

rst_prolog = """\
.. |full name| replace:: Open Seismic Risk Assessment Tool
.. |short name| replace:: OpenSRA
.. |github link| replace:: `OpenSRA Github page`_
.. _OpenSRA Github page: https://github.com/OpenSRA/OpenSRA
.. |messageBoard| replace:: `Message Board`_
.. |downloadLink| replace:: `OpenSRA download link`_
.. _OpenSRA download link: https://peer.berkeley.edu/opensra
.. |version| replace:: 0.1
.. |peer link| replace:: `PEER`_
.. _PEER: https://peer.berkeley.edu/
.. |slate link| replace:: `Slate Geotechnical Consultants`_
.. _Slate Geotechnical Consultants: http://slategeotech.com/
.. |ngawest2 link| replace:: `NGA West 2`_
.. _NGA West 2: https://peer.berkeley.edu/research/nga-west-2
.. |contact person| replace:: Barry Zheng, Slate Geotechnical Consultants, bzheng@slategeotech.com
.. |developers| replace:: Barry Zheng (Slate), 
                          Jennie Watson-Lamprey (Slate), 
                          Wael Elhaddad (SimCenter, NHERI), 
                          Frank McKenna (SimCenter, NHERI), 
"""


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
			  'sphinx.ext.todo',
			  'sphinx.ext.githubpages',
			  'numpydoc']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/OpenSRA_logo_1600ppi.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
html_theme = 'nature'