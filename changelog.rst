=========
Changelog
=========

- Unreleased
  - Fixed a compilation regression in csmat_binop
  - Bump MSRV (1.64)
  - add support for reading/writing Complex{32,64} matrices in matrixmarket format.
    Also change semantics so that files of kind  {integer,real,complex} are only readable
    into matrices of the same kind (so integer can be read into [iu]{8,16,32,64,size}),
    real into u{32,64}, complex into Complex{32,64}.  And raise a meaningful error if
    the user tries something that produces a mismatch.

    Also make skew-symmetric and symmetric matrices work for complex values.
  - add support for reading hermitian complex matrices in matrixmarket format.
    Of course, reading non-complex hermitian matrices is nonsensical, so this
    raises an error (on the off-chance that some other matrixmarket writer
    can write such matrices).

- 0.11.0
  - ``MulAcc`` is generalised to allow different output types from input
  - Bump `ndarray` to `0.15`. This requires a bump in MSRV to `1.49`

- 0.10.0
  - support more scalar types for scalar/matrix multiplication
  - refactor the handling of ``CsMatBase``'s ``indptr`` member to be able to
    express correct slicing views over the outer dimension **breaking change**
  - refactor slicing to get a nicer API **breaking change**
  - use more clippy lints to get more idiomatic code
  - add a ``diag`` method to ``CsMatBase``
  - fix a bug in ``CsMatViewMut::outer_iterator_mut``
  - add ``CsVecBase::to_dense``
  - split the error type between structural and linalg errors **breaking change**
  - allow specification of the ``Iptr`` type when converting from a triplet
    matrix to a compressed matrix **breaking change**
  - refactor matrix constructors for consistency **breaking change**
  - support generic dense vectors in solvers, {matrix,permutation}/vector
    multiplication. Technically a **breaking change**, but should not be in
    practice
  - remove need for `Copy` bound in most places
- 0.9.3
  - mitigate bugs in ``middle_outer_views``, that will require breaking changes
    in 0.10 to be fully fixed.
  - Add scalar mul for {u,i}{8,16}
  - Add CsMatBase::diag
- 0.9.2
    - Fix a crash on matrix products with 0 rows.
- 0.9.1
    - Fix a crash when parallelizing matrix products with less rows than the
      number of CPU cores.
- 0.9.0
    - Make FillInReduction enum non exhaustive to prevent excessive breakage
      when new algorithms are implemented. **breaking change**
    - Make rayon optional **breaking change**
    - Make fill in reduction enum non exhaustive **breaking change**
    - Add SuiteSparse's CAMD in the fill in reduction enum
    - Add structure only sparse matrix type
    - Add Kronecker product
    - Implemnt Approx for matrix comparison
    - Various performance improvements with code patterns enabling removal
      of bounds checking by the compiler
    - Improve performance of sparse matrix creation by checking if indices are
      sorted and sorting only if necessary for owning matrices.
    - Fix CSC/CSR multiplication which would fail without reason.
- 0.8.1
    - Expose the ``num_kinds`` module to allow generic usage of matrix market
      serialization functions in client crates
- 0.8.0
    - accelerate sparse matrix product, remove the old implementation
      **breaking change**
    - introduce fill-in reduction permutation using reverse Cuthil-McKee
    - fix permuted iteration that caused bugs in ``sprs-ldl``
      **breaking change**
    - check permutation validity on creation **breaking change**
- 0.7.1
    - fix issue when building docs on nightly, which broke on docs.sprs
- 0.7.0
    - make serde optional **breaking change**
    - make mul_acc_mat_vec_cs{r|c} more generic **breaking change**
    - support having different types for indptr & indices in CsMatBase **breaking change**
    - more careful overflow checking
    - upgrade dependencies
- 0.6.5
    - faster triplet format to compressed storage conversion
    - fix borrow checker issue flagged by new NLL
    - can read Matrix Market files from an ``io::BufRead``
    - improve ``CsMat::map`` to enable changing the storage type
- 0.6.4
    - add specialized sparse/sparse vector dot product using binary search
      for vectors where the number of non-zeros is very different.
    - enhance performance of sparse/sparse vector dot product
- 0.6.3
    - enforce rustfmt style checking
    - more explicit error messages when checking the structure of a ``CsMat``
    - ``into_raw_storage`` allows recylcling the storage of a ``CsMat``
    - support more ndarray versions
    - initial ``serde`` support
    - add more genericity over index type for ``CsMat`` construction functions.
    - ``CsMatBase`` now derives ``Clone``.
- 0.6.2
    - add support for symmetric matrices in Matrix Market IO
    - fix bug with adjacent empty columns in CSR matrix iteration.
- 0.6.1
    - fix ``to_dense`` for non-square matrices
    - improve performance of sparse-dense dot products and matrix vector
      products.
    - add support for Matrix Market IO.
- 0.6.0
    - enable the selection of the integer type for storing indices in matrix
      types **breaking change**
    - update to ndarray 0.10
    - refactor triplet matrix to use the same idioms used in compressed matrix
      **breaking change**
    - enhance documentation of main data structures
    - permutation constructor for identity permutation now requires the dimension
      on which the permutation should operate **breaking change**
- 0.5.0
    - adapt to breaking changes in ndarray 0.7
- 0.4.1:
    - add ``insert()`` method to insert an element inside an owned csmat
    - add ``outer_iterator_mut()`` method to enable changing the non-zero
      values of a sparse matrix while keeping its structure constant.
    - remove unsafe usage in the library
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

