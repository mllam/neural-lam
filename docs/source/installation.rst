Installation
============

Install from PyPI
------------------

For standard use:

.. code-block:: bash

   pip install neural_lam

Basic Installation (Development)
--------------------------------

Neural-LAM uses `uv <https://github.com/astral-sh/uv>`_ for dependency management.

Clone the repository:

.. code-block:: bash

   git clone https://github.com/mllam/neural-lam.git
   cd neural-lam

Install dependencies:

.. code-block:: bash

   uv sync

Note
----

Neural-LAM depends on several scientific and machine learning
libraries (e.g., PyTorch, Cartopy, Dask). Depending on your use
case (training, visualization, data preparation), additional
dependencies may be required.

Refer to ``pyproject.toml`` for the full dependency list.

Building the Documentation
--------------------------

To build the documentation locally, ensure the required documentation tools are installed (e.g., Sphinx, Furo, myst-parser).

Then run:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build

The generated HTML files will be available in:

``docs/build/html/``