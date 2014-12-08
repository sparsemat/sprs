CSRust, a sparse matrix library written in Rust
===============================================

CSRust implements some sparse matrix data structures and linear algebra
algorithms.

License
-------

BSD license. See LICENSE.txt

Features
--------

Structures
..........

- CSR matrix, able to operate on borrowed data
- CSR matrix, able to operate on owned data
- CSC matrix, unified code with CSR

TODO
----

Structures
..........

- CSC/CSR tests with more trickier shapes/data
- unified interface for owned and borrowed data
- lower/upper triangular CSC/CSR matrices
- block-sparse matrices

Algorithms
..........

- sparse triangular solve
- sparse LU decomposition
- sparse Cholesky decomposition

Misc
....

- Python bindings
