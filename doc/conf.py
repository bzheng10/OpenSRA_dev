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
import os, sys, json
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), 'src'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'src'))

# -- Preprocess/setup ---------------------------------------------------
# setup
cwd = os.getcwd()
opensra_dir = os.path.dirname(cwd)
# get list of devs from devs.json
with open(os.path.join(opensra_dir,'readme_src','ReadmeResource.json'),'r') as f:
    readme_resource = json.load(f)
devs = readme_resource['Developers'] # developers
depends = readme_resource['PythonModules'] # dependencies
ackns = readme_resource['Acknowledgements'] # acknowledgements
contacts = readme_resource['Contacts'] # contacts
py_ver = readme_resource['Python']['Version'] # Python version


# -- Project information -----------------------------------------------------

project = 'OpenSRA'
copyright = '2021, Slate Geotechnical Consultants and The Regents of the University of California'
author = 'Barry Zheng'

# The full version, including alpha/beta/rc tags
release = '(2022)'


# -- Keywords -----------------------------------------------------

rst_prolog = f"""\
.. |full name| replace:: Open Seismic Risk Assessment Tool
.. |short name| replace:: OpenSRA
.. |github link| replace:: `OpenSRA Github page`_
.. _OpenSRA Github page: https://github.com/OpenSRA/OpenSRA
.. |downloadLink| replace:: `OpenSRA download link`_
.. _OpenSRA download link: https://peer.berkeley.edu/opensra
.. |version| replace:: 0.4
.. |peer link| replace:: `PEER`_
.. _PEER: https://peer.berkeley.edu/
.. |slate link| replace:: `Slate Geotechnical Consultants (Slate)`_
.. _Slate Geotechnical Consultants (Slate): http://slategeotech.com/
.. |ngawest2 link| replace:: `NGA West 2`_
.. _NGA West 2: https://peer.berkeley.edu/research/nga-west-2
"""

# List of contacts
for count,item in enumerate(contacts):
    info = contacts[item]
    rst_prolog = rst_prolog + \
f"""\
.. |contact{count+1}| replace:: {item}, {info['Title']} ({info['Affiliation']}): {info['Profile']}
"""

# List of developers
for count,item in enumerate(devs):
    info = devs[item]
    rst_prolog = rst_prolog + \
f"""\
.. |dev{count+1}| replace:: {item}, {info['Title']} ({info['Affiliation']})
"""

# List of project managers
for count,item in enumerate(pms):
    info = pms[item]
    rst_prolog = rst_prolog + \
f"""\
.. |pm{count+1}| replace:: {item}, {info['Title']} ({info['Affiliation']})
"""

# List of acknowledgements
for count,item in enumerate(ackns):
    info = ackns[item]
    rst_prolog = rst_prolog + \
f"""\
.. |ackn{count+1}| replace:: {item}, {info['Title']} ({info['Affiliation']})
"""

# .. |messageBoard| replace:: `Message Board`_
# .. |contact1| replace:: Barry Zheng (Slate): bzheng@slategeotech.com
# .. |contact2| replace:: Jennie Watson-Lamprey (Slate): jwatsonlamprey@slategeotech.com
# .. |dev1| replace:: Barry Zheng (Slate)
# .. |dev2| replace:: Steve Gavrilovic (SimCenter, NHERI)
# .. |dev3| replace:: Frank McKenna (SimCenter, NHERI)
# .. |dev4| replace:: Maxime Lacour (UC Berkeley)
# .. |ackn1| replace:: Wael Elhaddad (SimCenter, NHERI)
# .. |ackn2| replace:: Kuanshi Zhong (SimCenter, NHERI)
# .. |ackn3| replace:: Simon Kwong (USGS)
# """

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
# html_theme = 'nature'
# html_theme = 'alabaster'
html_theme = 'classic'
# html_theme = 'furo'
# html_theme = 'press' # not working, compatibility with logo
# html_theme = 'insegel'
# html_theme = 'sphinx_material'