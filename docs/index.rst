``braincell`` documentation
===========================

`braincell <https://github.com/chaobrain/braincell>`_ provides dendritic modeling capabilities in JAX for brain dynamics.




----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U braincell[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U braincell[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U braincell[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


----


See also the brain modeling ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


We are building the `brain modeling ecosystem <https://brain-modeling.readthedocs.io/>`_.




.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation

   apis/changelog.md
   apis/braincell.rst
   apis/braincell.neuron.rst
   apis/braincell.synapse.rst
   apis/braincell.ion.rst
   apis/braincell.channel.rst
   apis/integration.rst



