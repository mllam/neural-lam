.. neural-lam documentation master file

MLLAM Autodoc Proof-of-Concept
==============================

This document demonstrates Autodoc's runtime introspection capabilities. 
Unlike static analysis tools, Autodoc correctly captures PyTorch Lightning 
lifecycle hooks and ``@property`` decorators in this codebase.

ARModel (Lifecycle Hooks Introspection)
---------------------------------------
.. autoclass:: neural_lam.models.ar_model.ARModel
   :members:
   :undoc-members:
   :show-inheritance:

Datastore Abstraction (@property Capture)
-----------------------------------------
.. autoclass:: neural_lam.datastore.base.BaseDatastore
   :members:
   :undoc-members:
   :show-inheritance: