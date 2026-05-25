``braincell.io`` module
=======================

.. currentmodule:: braincell.io
.. automodule:: braincell.io

``braincell.io`` reads neuronal morphologies from the common file formats and
integrates with `NeuroMorpho.Org <https://neuromorpho.org>`_. See
:doc:`../file_formats/index` for the task-oriented guide.


SWC
---

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    SwcReader
    SwcReadOptions
    SwcReport
    SwcIssue


Neurolucida ASC
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    AscReader
    AscReport
    AscIssue
    AscMetadata
    AscSpineRecord


NeuroML2
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    NeuroMlReader


NeuroMorpho.Org client
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    load_neuromorpho
    fetch_neuromorpho

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    NeuroMorphoClient
    NeuroMorphoQuery
    NeuroMorphoNeuron
    NeuroMorphoMeasurement
    NeuroMorphoDetail
    NeuroMorphoFilePlan
    NeuroMorphoDownloadItem
    NeuroMorphoDownloadRecord
    NeuroMorphoUrls
    NeuroMorphoSearchPage
    NeuroMorphoCache
    NeuroMorphoCacheLayout
    NeuroMorphoCacheStatus


Errors
------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    NeuroMorphoError
    NeuroMorphoHTTPError
    NeuroMorphoNotFoundError
    CheckpointError
    CheckpointVersionError


Checkpointing
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    save_morpho
    load_morpho
    save_branch
    load_branch
