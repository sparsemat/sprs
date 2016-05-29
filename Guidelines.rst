==========
Guidelines
==========

This document describes development guidelines for this library.

Error policy
============

As encouraged by rust, contract violations should be handled by panicking,
whereas softer errors such as I/O errors should be dealt with using the
``Result`` type. Here we describe, in the particular context of sparse linear
algebra, what we consider to be contract violations. Any error not mentionned
here should be dealt with using the ``Result`` type.

Contract violation
------------------

- *Dimension mismatch* in linear algebra operations
- *Storage assumption violation* in functions that require a specific storage
  type.
- *Out of bounds indices* when constructing a sparse matrix. Generally speaking
  all indices should be in bounds for the prescriber shape.
- *Length mismatch in constructors*, such as an ``indptr`` length not
  corresponding to the matrix' dimension.
- *Wrong workspace length*
