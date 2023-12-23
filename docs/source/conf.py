import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SQuADDS'
copyright = '2023, Sadman Ahmed Shanto & Eli Levenson-Falk'
author = 'Sadman Ahmed Shanto'
release = '0.1.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# Sphinx should ignore these patterns when building.
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "jupyter_execute",
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
]

autodoc_typehints = "none"
nbsphinx_execute = 'never'
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'qiskit_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "logo_only": True,
    "disable_ecosystem_logo": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
}

html_context = {
    "analytics_enabled": True,
    "expandable_sidebar": True,
    "theme_announcement": "🎉 Our paper is out!",
    "announcement_url": "https://arxiv.org/pdf/2312.13483.pdf",
    "announcement_url_text": "Check it out",
}

html_last_updated_fmt = "2023/12/22"
html_title = f"{project} {release}"


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
