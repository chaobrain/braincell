``braincell.morph`` module
==========================

.. currentmodule:: braincell.morph
.. automodule:: braincell.morph



`braincell.morph` provides diverse tools for morphology processing, enabling the representation
and manipulation of neuronal structures.

:class:`~braincell.Morphology`, :class:`~braincell.Branch` and the six branch types
(:class:`~braincell.Soma`, :class:`~braincell.Dendrite`, :class:`~braincell.Axon`,
:class:`~braincell.BasalDendrite`, :class:`~braincell.ApicalDendrite`,
:class:`~braincell.CustomBranch`) can also be imported from ``braincell.morph``;
``braincell.morph.Morphology is braincell.Morphology``. They are documented on the
:doc:`braincell` page.



Morphology Representation
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    MorphoBranch
    MorphoEdge
    MorphoMetric
    PrincipalAxes


Helpers
-------

.. autofunction:: branch_class_for_type
