import os
import re
import sys

# Set the path to the root of the project
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'SQuADDS'
copyright = '2026, Sadman Ahmed Shanto & Eli Levenson-Falk'
author = 'Sadman Ahmed Shanto'
version = "0.5.0"
release = "0.5.0"

# General configuration
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "jupyter_execute",
    "setup.py",
    "README.md",
    "imports_test.py",
    "CONTRIBUTING.md",
    "_extra",
]

# Standalone pages copied as-is into the built site: _extra/qcap/ -> <site>/qcap/ (the QCap paper page; /ml-paper/ redirects there) (the paper's explainer videos)
html_extra_path = ["_extra"]

extensions = [
    'sphinx.ext.autodoc',
    'nbsphinx',
    "sphinx.ext.extlinks",
    'qiskit_sphinx_theme',
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",  # Use only mathjax for math rendering
    "sphinx.ext.viewcode",
    "jupyter_sphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "reno.sphinxext",
    'myst_parser',  # Use MyST-Parser for Markdown support
]

# MyST-Parser configuration (optional, but can be extended)
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
]

# Let Sphinx autodetect .rst and .md
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Templates path
templates_path = ['_templates']

# HTML output configuration
html_logo = "./_static/images/squadds_logo_dark.png"
html_favicon = "./_static/images/squadds_logo_dark.png"

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
# "Paper" look: the docs set like the paper (STIX Two Text, running head, outline sidebar, journal tables)
html_css_files = ["css/paper.css"]
html_js_files = ["js/paper.js"]
html_theme_options = {
    "logo": {
        "text": "SQuADDS",
        "alt_text": "",
        "image_light": "_static/images/squadds_mark_light.png",
        "image_dark": "_static/images/squadds_mark_dark.png",
        "link": "https://lfl-lab.github.io/SQuADDS/"
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/LFL-Lab/SQuADDS",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "SQuADDS datasets on Hugging Face",
            "url": "https://huggingface.co/collections/SQuADDS/squadds-datasets-6a65152b857f8efa5620880a",
            "icon": "_static/images/hf-logo.svg",
            "type": "local",
        },
        {
            "name": "SQuADDS ML inference API",
            "url": "https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api",
            "icon": "fa-solid fa-code",
        },
        {
            "name": "Chat with SQuADDS",
            "url": "https://deepwiki.com/LFL-Lab/SQuADDS",
            "icon": "fa-regular fa-comments",
        },
    ],
    "navbar_start": ["paper-logo"],
    "navbar_center": ["paper-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button-field"],
    "article_header_start": ["paper-crumbs"],
    "primary_sidebar_end": [],
    "secondary_sidebar_items": {"source/squadds*": ["page-toc"], "source/modules": ["page-toc"]},
    "footer_start": ["paper-footer"],
    "footer_center": [],
    "footer_end": [],
    "navigation_depth": 5,
    "show_nav_level": 3,
    "collapse_navigation": True,
}
html_sidebars = {
    "index": [],
    "**": ["paper-outline"],
}
html_context = {
    "theme_announcement": "Preprint coming very soon. Check out our <a href='https://lfl-lab.github.io/SQuADDS/qcap/'>docsite</a> now",
    "expandable_sidebar": True,
    "google_analytics_id": "G-R5QKJDWM2W",
    "footer": "_footer.html",  # Add this line to reference your custom footer
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
nitpick_ignore = [
    ("py:class", "numpy.ndarray"),
    ("py:class", "np.ndarray"),
    ("py:class", "pathlib.Path"),
    ("py:class", "Path"),
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "squadds.ml.capacitance.models._Registry"),
    ("py:class", "squadds.calcs.floating_transmon._MaxwellTransmonHamiltonian"),
    ("py:class", "qiskit_metal.qlibrary.core.QComponent"),
    ("py:class", "qiskit_metal.qlibrary.qubits.transmon_cross.TransmonCross"),
]

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
    "keywords": "qiskit, qiskit-metal, qubit, Transmon, design, ansys, hfss, KLayout, superconducting, quantum, computing, SQuADDS, IBM, CPW, Hamiltonian, Sadman Ahmed Shanto, Eli Levenson-Falk, LFL Lab, USC Physics, USC, University of Southern California",
}


# Left outline ("Paper" look): the site's contents like PDF bookmarks, with the current page opened to its sections
_OUTLINE_SKIP = ("source/squadds", "source/modules", "genindex", "py-modindex", "search")


def _paper_outline(app, pagename, templatename, context, doctree):
    def outline():
        nav = str(
            context["generate_toctree_html"](
                "sidebar", startdepth=0, show_nav_level=1, maxdepth=5, collapse=True, includehidden=True, titles_only=True
            )
        )
        local = context.get("toc", "")
        if pagename.startswith(_OUTLINE_SKIP) or local.count("<li") < 2:
            return nav
        return re.sub(
            r'(<a class="current reference internal" href="#">.*?</a>)',
            lambda m: m.group(1) + '<div class="paper-local">' + local + "</div>",
            nav,
            count=1,
            flags=re.S,
        )

    context["paper_outline"] = outline


def setup(app):
    app.connect("html-page-context", _paper_outline)
