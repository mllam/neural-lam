"""Patch AutoAPI for astroid>=4."""  # codespell:ignore astroid

from __future__ import annotations

# Standard library
import inspect

# Third-party
from astroid import builder as astroid_builder  # codespell:ignore astroid
from astroid.manager import AstroidManager  # codespell:ignore astroid


def setup(app):
    builder_init = astroid_builder.AstroidBuilder.__init__
    if "manager" not in inspect.signature(builder_init).parameters:
        return {"version": "0.1"}
    original_builder = astroid_builder.AstroidBuilder

    class AutoapiAstroidBuilder(original_builder):
        def __init__(self, *args, **kwargs):
            if not args and "manager" not in kwargs:
                kwargs["manager"] = AstroidManager()
            super().__init__(*args, **kwargs)

    astroid_builder.AstroidBuilder = AutoapiAstroidBuilder
    return {"version": "0.1"}
