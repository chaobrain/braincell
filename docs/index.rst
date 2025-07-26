``braincell`` documentation
===========================

`braincell <https://github.com/chaobrain/braincell>`_ implements a biophysics-based State Transformation System for precise neuronal dynamics modeling and simulation.

``BrainCell`` is specifically designed to work with biologically detailed state-based models, including multi-compartment neurons with dendritic trees and ion channel dynamics.

``BrainCell`` serves as a core component in building a `Brain Dynamics Programming (BDP) ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_, bridging computational neuroscience and neuroengineering by enabling accurate simulations of neural dynamics at multiple scales.






----
Features
^^^^^^^^^

.. grid::



   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Biophysical State Precision
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``BrainCell`` enables biophysically accurate modeling of neural dynamics across scales, from ion channel gating to network-wide population activity, with state variables directly mapped to measurable biological quantities like membrane potential and ion concentrations.


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Stiff Dynamics Optimization
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``BrainCell`` features specialized solvers optimized for stiff neural systems, efficiently handling rapid biophysical transitions while integrating multi-compartment neuron structures.



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
   :maxdepth: 1
   :caption: Quickstart
   :hidden:

   quickstart/concepts-zh.ipynb





.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:


   tutorial/cell-zh.ipynb
   tutorial/channel-zh.ipynb
   tutorial/ion-zh.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Advanced Tutorials
   :hidden:

   advanced tutorial/rationale-zh.ipynb
   advanced tutorial/differential equation-zh.ipynb
   advanced tutorial/Examples-zh.ipynb
   advanced tutorial/more-zh.ipynb


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Documentation

   apis/changelog.md
   apis/braincell.rst
   apis/braincell.neuron.rst
   apis/braincell.synapse.rst
   apis/braincell.ion.rst
   apis/braincell.channel.rst

