"""salma package

This file ensures Python treats `projects.salma` as a regular package so
submodules like `projects.salma.utils.cma_warmup_hook` can be imported
reliably by MMEngine's custom_imports.
"""

# Optionally expose common subpackages
# from . import models, datasets, evaluation  # noqa: F401
