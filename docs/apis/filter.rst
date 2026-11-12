``braincell.filter`` module
===========================

.. currentmodule:: braincell.filter
.. automodule:: braincell.filter

``braincell.filter`` provides the selection algebra used to say *where* on a
cell a mechanism is installed. **Regions** select extended sets of cable (paint
targets); **locsets** select ordered, duplicate-preserving point rows (place
targets). Locsets concatenate through ``+`` and provide stable set operations
through ``|``, ``&``, and ``-``. See
:doc:`../concepts/regions_locsets` for the conceptual guide.


Regions
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    RegionExpr
    AllRegion
    EmptyRegion
    SubtreeRegion
    BranchRangeFilter
    BranchInFilter
    BranchSlice
    RadiusRangeRegion
    EuclideanDistanceRegion
    TreeDistanceRegion
    RegionAnchors
    RegionMask
    RegionSetOp


Locsets
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    LocsetExpr
    RootLocation
    Terminals
    ForkPoints
    UniformSamples
    StepSamples
    RandomSamples
    AtLocation
    LocsetMask
    LocsetConcatOp
    LocsetSetOp
    LocsetUniqueOp

``BranchPoints`` is an alias of :class:`ForkPoints`, kept for compatibility:
``braincell.filter.BranchPoints is braincell.filter.ForkPoints``.


Sampling and Batches
--------------------

:func:`sample` builds a lazy :class:`SampleLocations` expression that draws
random locations continuously over a region; a sampling-density callable
receives a :class:`SamplingContext`. A :class:`LocsetBatch` holds several
resolved locsets, aligned.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    LocsetBatch
    SampleLocations
    SamplingContext

.. autosummary::
   :toctree: generated/
   :nosignatures:

    sample


Helpers
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    branch_in
    branch_range
    at


Caching
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    SelectionCache
