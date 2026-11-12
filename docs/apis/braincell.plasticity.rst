``braincell.plasticity`` module
===============================

.. currentmodule:: braincell.plasticity
.. automodule:: braincell.plasticity

A rule changes the weight of every row of one ``connect`` call while the model
runs. Declare it with :class:`braincell.mech.ConnectionRule` and pass it as
``connect(..., plasticity=...)``; read the dynamic weight and the rule's other
states with :meth:`braincell.ConnectionView.rule_state`. A new rule subclasses
:class:`braincell.ConnectionRule` and is registered with
:func:`braincell.mech.register_connection_rule`.


Spike-Timing-Dependent Plasticity
---------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    PairSTDP
