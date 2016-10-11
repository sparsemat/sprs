sprs, sparse matrices for Rust
==============================

.. image:: https://travis-ci.org/vbarrielle/sprs.svg?branch=master
    :target: https://travis-ci.org/vbarrielle/sprs

sprs implements some sparse matrix data structures and linear algebra
algorithms in pure Rust.

The API is a work in progress, and feedback on its rough edges is highly
appreciated :)

Features
--------

Structures
..........

- CSR/CSC matrix
- triplet matrix
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
- simple sparse Cholesky decomposition (requires opting into an LGPL license)
- sparse triangular solves with dense right-hand side


Examples
--------

Matrix construction

.. code-block:: rust

  use sprs::{CsMat, CsMatOwned, CsVec};
  let eye : CsMatOwned<f64> = CsMat::eye(3);
  let a = CsMat::new_csc((3, 3),
                         vec![0, 2, 4, 5],
                         vec![0, 1, 0, 2, 2],
                         vec![1., 2., 3., 4., 5.]);

Matrix vector multiplication


.. code-block:: rust

  use sprs::{CsMat, CsVec};
  let eye = CsMat::eye(5);
  let x = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
  let y = &eye * &x;
  assert_eq!(x, y);

Matrix matrix multiplication, addition

.. code-block:: rust

  use sprs::{CsMat, CsVec};
  let eye = CsMat::eye(3);
  let a = CsMat::new_csc((3, 3),
                         vec![0, 2, 4, 5],
                         vec![0, 1, 0, 2, 2],
                         vec![1., 2., 3., 4., 5.]);
  let b = &eye * &a;
  assert_eq!(a, b.to_csr());

For a more comple example, be sure to check out the ``examples/`` folder.

Documentation
-------------

Documentation is available at docs.rs_.

.. _docs.rs: https://docs.rs/sprs

Changelog
---------

- next version:
    - add ``insert()`` method to insert an element inside an owned csmat
- 0.4.0:
    - panic for contract violations, use errors only for recoverable problems
      **breaking change**
    - depend on latest ndarray version: 0.6 **braking change**
    - refactor API to present shorter import paths **breaking change**
    - expose sparse matrix / dense vector product via ``Mul``.
    - add an example of building and solving a sparse linear system
- O.4.0-alpha.4 version, most changes are **breaking changes**:
    - move cholesky factorization into its own crate
    - add ``to_dense()`` method for sparse matrices
    - rename ``borrowed()`` into ``view()``
    - ``outer_iterator()`` no longer returns the index of the dimension we're
      iterating. The old behavior can be obtained by chaining a call
      to ``enumerate()``.
    - ``eye()`` returns a csr matrix by default, a csc matrix can be obtained
      using ``eye_csc()``.
    - rename ``new_borrowed()`` into ``new_view()``.
    - rename ``new_raw()`` into ``new_view_raw()``.
    - rename ``new_owned()`` into ``new()`` or ``new_csc()`` depending on the
      desired ordering, and have the ownning constructors panic on bad input.
    - constructors now take a tuple for shape information
- 0.4.0-alpha.3:
    - rename ``at`` family of functions into ``get``, consistent with the naming
      scheme in standard library. **breaking change**
    - move cholesky factorization behind the "lgpl" feature flag
      **rbeaking change**
    - per-nnz-element function application (``map``, ``map_inplace``).
    - binary operations operating on matching non-zero elements
      (``csvec_binop``, ``csmat_binop``).
    - introduce ``nnz_index`` to retrieve an index of an element allowing
      for later constant time access.
- 0.4.0-alpha.2:
    - functions in the ``at`` family will return references **breaking change**
    - simpler arguments for ``at_outer_inner`` **breaking change**
    - mutable view types
- 0.4.0-alpha.1:
    - depend on ndarray for dense matrices **breaking change**
    - iterators return reference where possible **breaking change**
    - remove unnecessary copy bounds
    - constructors to build sparse matrices from dense matrices
    - forward some LdlSymbolic methods in LdlNumeric
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

