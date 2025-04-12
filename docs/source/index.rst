Welcome to PassivePy's documentation!
===================================

PassivePy is a Python package for passive voice detection and analysis in text. It provides robust tools for identifying passive voice constructions and analyzing their usage patterns.

Features
--------

* Multiple detection methods (regex and SpaCy-based)
* Configurable detection parameters
* Detailed analysis of passive voice patterns
* Support for batch processing
* Comprehensive logging and error handling

Installation
-----------

.. code-block:: bash

   pip install passivepy

Quick Start
----------

.. code-block:: python

   from passivepy import PassiveDetector

   # Initialize detector
   detector = PassiveDetector()

   # Detect passive voice
   text = "The book was written by the author."
   result = detector.detect(text)

   print(f"Is passive: {result['is_passive']}")
   print(f"Passive phrases: {result['passive_phrases']}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   examples
   contributing
   changelog

Indices and tables
-----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 