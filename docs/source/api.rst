API Reference
============

This page provides detailed documentation for the PassivePy API.

Core Module
----------

.. automodule:: passivepy.core.passive_voice_detector
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: passivepy.core.types
   :members:
   :undoc-members:
   :show-inheritance:

Utils Module
-----------

.. automodule:: passivepy.utils.logging
   :members:
   :undoc-members:
   :show-inheritance:

Analysis Module
-------------

.. automodule:: passivepy.analysis
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from passivepy import PassiveDetector

   detector = PassiveDetector()
   result = detector.detect("The book was written by the author.")
   print(result)

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from passivepy import PassiveDetector
   from passivepy.core.types import DetectorConfig

   config = DetectorConfig(
       threshold=0.8,
       use_spacy=True,
       language="en",
   )
   detector = PassiveDetector(config)
   result = detector.detect("The project was completed on time.")
   print(result)

Batch Processing
~~~~~~~~~~~~~~

.. code-block:: python

   from passivepy import PassiveDetector

   detector = PassiveDetector()
   texts = [
       "The book was written by the author.",
       "The project was completed on time.",
       "The results were presented to the committee.",
   ]
   results = [detector.detect(text) for text in texts]
   print(results) 