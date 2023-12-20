import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SQuADDS'
copyright = '2023, Sadman Ahmed Shanto & Eli Levenson-Falk'
author = 'Sadman Ahmed Shanto'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'nbsphinx',
    'qiskit_sphinx_theme',
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
    "disable_ecosystem_logo": True,
}

html_context = {
    "theme_announcement": "🎉 Our paper is out!",
    "announcement_url": "https://example.com",
    "announcement_url_text": "Check it out",
}
