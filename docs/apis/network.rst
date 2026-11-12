``braincell.network`` module
============================

.. currentmodule:: braincell.network

A :class:`~braincell.Network` holds named populations of cells and event
sources, connects them, and runs them together. These names are documented
under the ``braincell`` top level, where they are also exported;
``braincell.network.Network is braincell.Network``.

.. currentmodule:: braincell


Networks
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    Network
    NetworkConnections
    NetworkResult


Connections
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    ConnectionView

.. autosummary::
   :toctree: generated/
   :nosignatures:

    connect


Event Sources
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    EventSource
    EventSourceView
    EventSequence
    EventTable
    NetStim
    VoltageCrossingSource


Recording
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    RecordingSpec
    RecordingSchema
    SampleBlock
    EventSeries

.. autosummary::
   :toctree: generated/
   :nosignatures:

    observe


Populations
-----------

.. currentmodule:: braincell.network

:class:`Population` is exported from ``braincell.network`` only.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    Population
