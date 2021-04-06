.. Sketch documentation master file, created by
   sphinx-quickstart on Thu Apr  1 14:51:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Sketch's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Features
========
1. Bloom Filter
2. HyperLogLog
3. SetSketch
4. Fast Hamming space distance functions
5. ngram hashing code


Modules
========
There are separate modules for each sketch structure for which there are bindings.
 * sketch.hll, providing HyperLogLog and comparison, and serialization functions
 * sketch.bf, providing Bloom Filters and comparison, and serialization functions
 * sketch.bbmh, providing b-bit minhash implementation + comparison, and serialization functions
 * sketch.setsketch, providing set sketch + comparison, and serialization functions


For each of these, the module provides construction - either taking parameters or a path to a file.
Each of these can be written to and read from a file with .write() and a constructor.
They can be compared with each other with member functions, or you can calculate comparison matrices via
`sketch.util.jaccard\_matrix`, `sketch.util.containment\_matrix`, `sketch.util.union\_size\_matrix`, `sketch.util.intersection_matrix`, all of which are in the util module.

Additionally, there are utilities for pairwise distance calculation in the `util` module.


Additional utilities: sketch.util
=====================

* fastdiv/fastmod:
    * Python bindings for fastdiv/fastmod; See https://arxiv.org/abs/1902.01961
    * fastdiv\_ and fastmod\_ are in-place modifications, while the un-suffixed returns a new array
* count\_eq
    ** Compute # of equal registers between two 1-d numpy arrays.
* pcount\_eq
    ** Compute row-pair-wise equal register counts between two 2-d numpy arrays.
* shs\isz
    ** Computes intersection size between two sorted hash sets.
* hash
    ** hashes strings

Python-only Code
===============

.. literalinclude:: ../../python/sketch/__init__.py
   :language: python
   :linenos:
   :lines: 11-25

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
