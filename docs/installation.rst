Installation
=============

Requirements
------------

* Python >= 3.10
* CUDA-compatible GPU (recommended) or CPU

PyPI Installation
-----------------

The easiest way to install Neural-LAM is via pip:

.. code-block:: bash

   python -m pip install neural-lam

Developer Installation
----------------------

To install from source for development:

Using ``uv`` (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/mllam/neural-lam.git
   cd neural-lam

2. Install ``uv``:

.. code-block:: bash

   pip install uv

3. Create a virtual environment (optional, recommended):

.. code-block:: bash

   uv venv --no-project

4. Install dependencies. For CPU-only version:

.. code-block:: bash

   uv pip install torch --index-url https://download.pytorch.org/whl/cpu

   For GPU with CUDA 12.1:

.. code-block:: bash

   uv pip install torch --index-url https://download.pytorch.org/whl/cu121

5. Install neural-lam in development mode:

.. code-block:: bash

   uv pip install --group dev -e .

Using ``pip``
~~~~~~~~~~~~~

1. Clone and navigate to repository:

.. code-block:: bash

   git clone https://github.com/mllam/neural-lam.git
   cd neural-lam

2. Create a virtual environment (recommended):

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

3. Install PyTorch. For CPU-only:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

   For GPU with CUDA:

.. code-block:: bash

   pip install torch

4. Install neural-lam and development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Torch Installation Notes
------------------------

PyTorch creates different package variants for different CUDA versions and CPU-only support. You may need to install torch separately based on your system configuration:

* **CPU-only**: ``pip install torch --index-url https://download.pytorch.org/whl/cpu``
* **CUDA 12.1**: ``pip install torch --index-url https://download.pytorch.org/whl/cu121``
* **Latest GPU**: ``pip install torch`` (requires latest CUDA)

See the `PyTorch install page <https://pytorch.org/get-started/locally/>`_ for more options.

Verifying Installation
----------------------

To verify your installation:

.. code-block:: python

   import neural_lam
   print(neural_lam.__version__)

Run the test suite:

.. code-block:: bash

   pytest tests/
