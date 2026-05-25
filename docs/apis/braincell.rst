``braincell`` module
====================

.. currentmodule:: braincell
.. automodule:: braincell












Base Class for Cell Modeling
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    HHTypedNeuron
    SingleCompartment
    Cell
    RunResult



Base Class for Ion Channels
---------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    IonChannel
    Ion
    IonInfo
    MixIons
    Channel


Ion Helpers
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    mix_ions


Discretization: Control Volumes and Policies
---------------------------------------------

Static, declaration-time representation of how a multi-compartment cell is
divided into control volumes (CVs), plus the policies that decide how many
CVs each branch receives.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    CV
    CVTree
    Node
    NodeTree
    CVPolicy
    CVPerBranch
    CVPolicyByTypeRule
    CompositeByTypePolicy
    DLambda
    MaxCVLen


Morphology
----------

Neuronal morphology container and the branch types used to label sections of
a reconstruction.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    Morphology
    Branch
    Soma
    Dendrite
    Axon
    BasalDendrite
    ApicalDendrite
    CustomBranch


Stimulus and Cable Mechanisms
-----------------------------

Passive cable properties and stimulus clamps attachable to a cell. See
:doc:`braincell.mech` for the full declarative mechanism layer.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    CableProperty
    CurrentClamp
    SineClamp
    FunctionClamp
