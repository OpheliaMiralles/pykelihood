# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "pykelihood"
copyright = "2025, Ophélia Miralles"
author = "Ophélia Miralles"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []
add_module_names = False
autosummary_generate = True
autodoc_typehints = "none"
autodoc_default_options = {
    "inherited-members": None,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_css_files = ["custom.css"]
html_static_path = ["_static"]
html_logo = "_static/logo.png"

html_theme_options = {
    "show_nav_level": 2,
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pykelihood/pykelihood",
            "icon": "fa-brands fa-github",
        }
    ],
    "navbar_align": "content",
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/darklogo.png",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["theme-switcher"],
    "navbar_end": ["search-field.html"],
    "secondary_sidebar_items": [],
}
html_context = {
    "github_url": "https://github.com",
    "github_user": "OpheliaMiralles",
    "github_repo": "pykelihood",
}
html_sidebars = {
    "**": ["globaltoc.html", "sourcelink.html"],
}
intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
autodoc_member_order = "alphabetical"


def skip_class_or_static_methods(app, what, name, obj, skip, options):
    # Skip classmethods and staticmethods (optional: add more logic if needed)
    if isinstance(obj, (classmethod, staticmethod)):
        return True
    if name.startswith("_") and not name.startswith("__"):
        return True  # Skip "protected" members like _my_method
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_class_or_static_methods)
