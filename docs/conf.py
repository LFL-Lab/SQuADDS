import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'SQuADDS'
copyright = '2023, Sadman Ahmed Shanto & Eli Levenson-Falk'
author = 'Sadman Ahmed Shanto'
release = "0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# Sphinx should ignore these patterns when building.
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "jupyter_execute",
    "*/setup.py",
    "../setup.py",
    "../../setup.py",
    "setup.py",
    "README.md",
    "../README.md",
    "../imports_test.py",
    "../../imports_test.py",
    "imports_test.py",
]


extensions = [
    'sphinx.ext.autodoc',
    'nbsphinx',
    "sphinx.ext.extlinks",
    'qiskit_sphinx_theme',
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "jupyter_sphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.katex",
    "reno.sphinxext",
]

autodoc_typehints = "none"
nbsphinx_execute = 'never'
templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']


html_context = {
#    "analytics_enabled": True,
    "theme_announcement": "🎉 Our <a href='https://arxiv.org/pdf/2312.13483.pdf'>paper</a> is out! ",
    "expandable_sidebar": True,
}

html_theme_options = {
  #  "logo_only": True,
  #  "disable_ecosystem_logo": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "github_url": "https://github.com/LFL-Lab/SQuADDS",
    "icon_links": [
        {
            "name": "LFL Lab",
            "url": "https://dornsife.usc.edu/lfl/",
            "icon": "fas fa-hand-peace",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/SQuADDS/",
            "icon": "fab fa-python",
        },
    ],
    "navigation_depth": 5,
    "show_nav_level": 3,
    "collapse_navigation": True,
}




html_last_updated_fmt = "%Y/%m/%d"
html_title = f"{project} v{release}"

add_module_names = True
modindex_common_prefix = ["squadds."]
autodoc_mock_imports = ["qutip", "scqubits", "pyaedt", "qiskit-metal", "setup"]

autodoc_typehints = "description"
# Only add type hints from signature to description body if the parameter has documentation.  The
# return type is always added to the description (if in the signature).
autodoc_typehints_description_target = "documented_params"

autoclass_content = "both"

autosummary_generate = True
autosummary_generate_overwrite = False

add_module_names = True

# This allows RST files to put `|version|` in their file and
# have it updated with the release set in conf.py.
rst_prolog = f"""
.. |version| replace:: {release}
"""

# Options for autodoc. These reflect the values from Terra.
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"

# This adds numbers to the captions for figures, tables,
# and code blocks.
numfig = True
numfig_format = {"table": "Table %s"}

# Settings for Jupyter notebooks.
nbsphinx_thumbnails = {
    # Default image for thumbnails.
    "**": "_static/images/logo.png",
}
