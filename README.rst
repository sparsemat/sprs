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

- CSR/CSC matrix, able to operate on borrowed or owned data

Algorithms
..........

- Outer iterator on compressed sparse matrices
- CSC/dense vector product
- CSR/dense vector product

TODO
----

Structures
..........

- CSC/CSR tests with more trickier shapes/data
- lower/upper triangular CSC/CSR matrices
- block-sparse matrices

Algorithms
..........

- tests on the outer iterator
- tests on the CSC/vector product
- sparse triangular solve
- sparse LU decomposition
- sparse Cholesky decomposition

Misc
....

- Python bindings
