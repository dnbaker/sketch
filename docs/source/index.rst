.. Sketch documentation master file, created by
   sphinx-quickstart on Thu Apr  1 14:51:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation for Sketch!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents::


Installation
===========

One-line::
    git clone https://github.com/dnbaker/sketch && cd sketch/python && python3 setup.py build_ext -j4 && python3 setup.py install

Features
========
1. Sketch structure bindings:
    1. Bloom Filter
    2. HyperLogLog
    3. B-bit minhash
    4. Set Sketch
2. Distance calculation functions
3. Miscellaneous
    1. fastmod/fastdiv for integer reductions
    2. ngram hashing
    3. fast hamming space distance calculations


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
_sketch.util.jaccard\_matrix_, _sketch.util.containment\_matrix_, _sketch.util.union\_size\_matrix_, _sketch.util.intersection_matrix_, all of which are in the util module.

Additionally, there are utilities for pairwise distance calculation in the `util` module.


Additional utilities: sketch.util
=====================

* fastdiv/fastmod:
    * Python bindings for fastdiv/fastmod; See https://arxiv.org/abs/1902.01961
    * fastdiv\_ and fastmod\_ are in-place modifications, while the un-suffixed returns a new array
* count\_eq
    * Compute # of equal registers between two 1-d numpy arrays.
* pcount\_eq
    * Compute row-pair-wise equal register counts between two 2-d numpy arrays.
* shs\isz
    * Computes intersection size between two sorted hash sets.
* hash
    * hashes strings

Computing optimal a and b
------
For lossy compression via quantization, _optimal\_ab_ computes the parameter values for best using hash space.


.. literalinclude:: ../../python/sketch/__init__.py
   :language: python
   :linenos:
   :lines: 11-25

Indices and tables
------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
