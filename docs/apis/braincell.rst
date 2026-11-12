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
    Cell
    RunResult


Cell Views
----------

Views over selected members of a :class:`Cell` population (``cell[...]``
returns a :class:`CellView`) and over the logical channel, ion, clamp and
synapse instances installed on it.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    CellView
    ChannelView
    ClampView
    IonView
    SynapseView



Base Classes for Ion Channels and Synapses
------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    IonChannel
    Ion
    IonInfo
    MixIons
    Channel
    Synapse

:class:`Synapse` is the runtime base class of point synapses; the declaration a
cell places is :class:`braincell.mech.Synapse`.


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
    Lumped
    MaxCVLen
    CVPerBranchList
    PointPlacement


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


Version
-------

.. autodata:: __version__

.. autodata:: __version_info__
