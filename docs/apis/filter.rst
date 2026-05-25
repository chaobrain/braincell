``braincell.filter`` module
===========================

.. currentmodule:: braincell.filter
.. automodule:: braincell.filter

``braincell.filter`` provides the selection algebra used to say *where* on a
cell a mechanism is installed. **Regions** select extended sets of cable (paint
targets); **locsets** select sets of points (place targets). Both compose
through union / intersection / difference. See
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
    BranchPoints
    UniformSamples
    StepSamples
    RandomSamples
    AtLocation
    LocsetMask
    LocsetSetOp


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
