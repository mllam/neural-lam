Installation
============

Basic Installation
------------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/mllam/neural-lam.git
   cd neural-lam

Create a virtual environment and install in editable mode:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .

Note
----

Neural-LAM depends on several scientific and machine learning
libraries (e.g., PyTorch, Cartopy, Dask). Depending on your use
case (training, visualization, data preparation), additional
dependencies may be required.

Refer to ``pyproject.toml`` for the full dependency list.

Building the Documentation
--------------------------

To build the documentation locally:

.. code-block:: bash

   cd docs
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   make html

The generated HTML files will be available in:

``docs/build/html/``