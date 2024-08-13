import os
import sys

from recommonmark.parser import CommonMarkParser

# Set the path to the root of the project
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'SQuADDS'
copyright = '2023, Sadman Ahmed Shanto & Eli Levenson-Falk'
author = 'Sadman Ahmed Shanto'
release = "0.2.4"

# General configuration
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "jupyter_execute",
    "setup.py",
    "README.md",
    "imports_test.py",
    "CONTRIBUTING.md",
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
    'recommonmark', # Add this for Markdown support
]

# Add source suffix and parser configuration
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

source_parsers = {
    '.md': CommonMarkParser,
}

# Templates path
templates_path = ['_templates']

# HTML output configuration
html_logo = "./_static/images/squadds_logo_dark.png"
html_favicon = "./_static/images/squadds_logo_dark.png"

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "logo": {
        "alt_text": "SQuADDS Logo",
        "image_light": "_static/images/squadds_logo_light_name.png",
        "image_dark": "_static/images/squadds_logo_dark_name.png",
        "link": "https://lfl-lab.github.io/SQuADDS/"
    },
    "github_url": "https://github.com/LFL-Lab/SQuADDS",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/SQuADDS/",
            "icon": "fab fa-python",
        },
        {
            "name": "HuggingFace",
            "url": "https://huggingface.co/datasets/SQuADDS/SQuADDS_DB",
            "icon": "fas fa-face-smile",
        },
        # {
            # "name": "LFL Lab",
            # "url": "https://dornsife.usc.edu/lfl/",
            # "icon": "fas fa-hand-peace",
        # },
    ],
    "navigation_depth": 5,
    "show_nav_level": 3,
    "collapse_navigation": True,
}
html_context = {
    "theme_announcement": "🎉 Our <a href='https://arxiv.org/pdf/2312.13483.pdf'>paper</a> is out! ",
    "expandable_sidebar": True,
    "google_analytics_id": "G-R5QKJDWM2W", 
}

html_last_updated_fmt = "%Y/%m/%d"
html_title = f"{project} v{release}"

# Autodoc settings
autodoc_mock_imports = ["qutip", "scqubits", "pyaedt", "qiskit_metal", "setup", "imports_test"]
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autoclass_content = "both"

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = False

# nbsphinx settings
nbsphinx_execute = 'never'

# allow html
nbsphinx_allow_html = True

# Jupyter notebook settings
nbsphinx_thumbnails = {
    "**": "_static/images/squadds.svg",
}

# Add version to the docs
rst_prolog = f"""
.. |version| replace:: {release}
"""

# Enable nitpicky mode to warn about broken references
nitpicky = True

# Ensure that MathJax knows to look for LaTeX delimiters
mathjax_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
    'tex2jax': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    },
}

html_meta = {
    "description": "SQuADDS: A Python package for design and simulation of superconducting quantum devices",
    "keywords": "qiskit, qiskit-metal, qubit, Transmon, design, ansys, hfss, KLayout, superconducting, quantum, computing, SQuADDS, IBM, CPW, Hamiltonian",
}