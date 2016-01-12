sprs, sparse matrices for Rust
==============================

.. image:: https://travis-ci.org/vbarrielle/sprs.svg?branch=master
    :target: https://travis-ci.org/vbarrielle/sprs

sprs implements some sparse matrix data structures and linear algebra
algorithms.

WARNING: this library is still in development, its API is not stable yet.

Features
--------

Structures
..........

- CSR/CSC matrix
- Sparse vector

Operations
..........

- sparse matrix / sparse vector product
- sparse matrix / sparse matrix product
- sparse matrix / sparse matrix addition, subtraction
- sparse vector / sparse vector addition, subtraction, dot product
- sparse/dense matrix operations

Algorithms
..........

- Outer iterator on compressed sparse matrices
- sparse vector iteration
- sparse vectors joint non zero iterations
- simple sparse Cholesky decomposition
- sparse triangular solves with dense right-hand side


Examples
--------

Matrix construction

.. code-block:: rust

  use sprs::{CsMat, CsMatOwned, CsVec};
  let eye : CsMatOwned<f64> = CsMat::eye(sprs::CSR, 3);
  let a = CsMat::new_owned(sprs::CSC, 3, 3,
                           vec![0, 2, 4, 5],
                           vec![0, 1, 0, 2, 2],
                           vec![1., 2., 3., 4., 5.]).unwrap();

Matrix vector multiplication


.. code-block:: rust

  use sprs::{CsMat, CsVec};
  let eye = CsMat::eye(sprs::CSR, 5);
  let x = CsVec::new_owned(5, vec![0, 2, 4], vec![1., 2., 3.]).unwrap();
  let y = &eye * &x;
  assert_eq!(x, y);

Matrix matrix multiplication, addition

.. code-block:: rust

  use sprs::{CsMat, CsVec};
  let eye = CsMat::eye(sprs::CSR, 3);
  let a = CsMat::new_owned(sprs::CSC, 3, 3,
                           vec![0, 2, 4, 5],
                           vec![0, 1, 0, 2, 2],
                           vec![1., 2., 3., 4., 5.]).unwrap();
  let b = &eye * &a;
  assert_eq!(a, b.to_csr());

Documentation
-------------

https://vbarrielle.github.io/sprs/doc/sprs/

Changelog
---------

- 0.3.3
  - switch to dual MIT/Apache-2.0 license
- 0.3.2
  - triplet matrix format for easier initialization
- 0.3.1
  - trait to abstract over sparse vectors
- 0.3.0
  - LDLT decomposition with support for permutations
- 0.2.6
  - lifetime issue fixed (revealed by rust 1.4)
- 0.2.5
  - sparse triangular / sparse rhs solvers
- 0.2.4
  - sparse triangular / dense rhs solvers
  - avoid "*" in dependencies
- 0.2.3
  - initial support for sparse/dense matrix addition
- 0.2.2
  - initial support for sparse/dense matrix multiplication
- 0.2.1
  - remove type aliases from impl blocks (doc issue)
- 0.2.0
  - matrix multiplication, addition
  - block matrix constructors (vstack, hstack, bmat)
  - trait to abstract over sparse matrices
- 0.1.0
  - first release on crates.io

License
-------

Licensed under either of

 * Apache License, Version 2.0, (./LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license (./LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

Contribution
............

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

