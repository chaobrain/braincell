``braincell.trainable`` module
==============================

.. currentmodule:: braincell.trainable
.. automodule:: braincell.trainable


Parameter Sources
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    parameter
    parameterized
    scale

.. data:: ParameterSource

   The type of a parameter source: the union of the source types that
   :func:`parameter`, :func:`parameterized` and :func:`scale` return. Use it to
   annotate a value that may be any of them.


Managing Parameters
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    TrainableManager
    ParameterSet
    ParameterBinding
