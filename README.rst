sprs, a sparse matrix library written in Rust
=============================================

sprs implements some sparse matrix data structures and linear algebra
algorithms.

WARNING: experimental lib, neither API nor functionality stable

License
-------

BSD license. See LICENSE.txt

Features
--------

Structures
..........

- CSR/CSC matrix, able to operate on borrowed or owned data

Operations
..........

- sparse matrix vector product
- sparse matrix matrix product
- sparse matrix matrix addition

Algorithms
..........

- Outer iterator on compressed sparse matrices
- CSC/dense vector product
- CSR/dense vector product
- sparse Cholesky decomposition

TODO
----

Structures
..........

- CSC/CSR tests with more trickier shapes/data
- lower/upper triangular CSC/CSR matrices
- block-sparse matrices

Operations
..........

- rebind operations with the corresponding traits

Algorithms
..........

- tests on the outer iterator
- tests on the CSC/vector product
- sparse triangular solve
- sparse LU decomposition

Misc
....

- Python bindings


API guidelines
--------------

Each exposed functionality should, if deemed necessary for performance reasons,
be exposed as a low-level C-style function (similar to BLAS API), and also
exposed as a high level API with good defaults for the performance related
arguments.
