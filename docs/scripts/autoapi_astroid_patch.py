"""
Patch AutoAPI for astroid >= 4 compatibility.

This module patches the `AstroidBuilder` from `astroid` to pass a `manager`
argument to the parent class `__init__`, which is strictly required for
astroid >= 4. AutoAPI does not natively pass this argument in its current
versions, so this workaround prevents documentation build errors.
"""  # codespell:ignore astroid

from __future__ import annotations

# Standard library
import inspect

try:
    # Third-party
    from astroid import builder as astroid_builder  # codespell:ignore astroid
    from astroid.manager import AstroidManager  # codespell:ignore astroid

    ASTROID_AVAILABLE = True
except ImportError:
    ASTROID_AVAILABLE = False


def setup(app):
    if not ASTROID_AVAILABLE:
        return {"version": "0.1"}

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
