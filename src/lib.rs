/*!

sprs is a sparse linear algebra library for Rust.

It features a sparse matrix type, [**`CsMat`**](struct.CsMatBase.html), and a sparse vector type,
[**`CsVec`**](struct.CsVecBase.html), both based on the
[compressed storage scheme](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29).

## Features

- sparse matrix/sparse matrix addition, multiplication.
- sparse vector/sparse vector addition, dot product.
- sparse matrix/dense matrix addition, multiplication.
- sparse triangular solves.
- powerful iteration over the sparse structure, enabling easy extension of the library.
- matrix construction using the [triplet format](struct.TriMatBase.html),
  vertical and horizontal stacking, block construction.
- sparse cholesky solver in the separate crate `sprs-ldl`.
- fully generic integer type for the storage of indices, enabling compact
  representations.
- planned interoperability with existing sparse solvers such as SuiteSparse.

## Quick Examples

Matrix construction:

```rust
use sprs::{CsMat, CsVec};
let eye : CsMat<f64> = CsMat::eye(3);
let a = CsMat::new_csc((3, 3),
                       vec![0, 2, 4, 5],
                       vec![0, 1, 0, 2, 2],
                       vec![1., 2., 3., 4., 5.]);
```

Matrix vector multiplication:

```rust
use sprs::{CsMat, CsVec};
let eye = CsMat::eye(5);
let x = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
let y = &eye * &x;
assert_eq!(x, y);
```

Matrix matrix multiplication, addition:

```rust
use sprs::{CsMat, CsVec};
let eye = CsMat::eye(3);
let a = CsMat::new_csc((3, 3),
                       vec![0, 2, 4, 5],
                       vec![0, 1, 0, 2, 2],
                       vec![1., 2., 3., 4., 5.]);
let b = &eye * &a;
assert_eq!(a, b.to_csc());
```

*/

#![deny(warnings)]

extern crate num_traits;
extern crate ndarray;

mod sparse;
pub mod errors;
pub mod stack;
pub mod indexing;

/// Deprecated type alias, will be removed on next breaking change
pub type Ix_ = ndarray::Ix1;
pub type Ix1 = ndarray::Ix1;
pub type Ix2 = ndarray::Ix2;

pub use sparse::{
    CsMatBase,
    CsMat,
    CsMatI,
    CsMatView,
    CsMatViewI,
    CsMatViewMut,
    CsMatViewMutI,
    CsMatVecView,
    CsVecBase,
    CsVecView,
    CsVecViewI,
    CsVec,
    CsVecI,
    CsVecViewMut,
    CsVecViewMutI,
    TriMatBase,
    TriMat,
    TriMatView,
    TriMatViewMut,
    TriMatI,
    TriMatViewI,
    TriMatViewMutI,
};


pub use sparse::symmetric::{
    is_symmetric,
};

pub use sparse::permutation::{
    Permutation,
    PermView,
    PermViewI,
    PermOwned,
    PermOwnedI,
};

pub use sparse::CompressedStorage::{
    self,
    CSR,
    CSC,
};

pub use sparse::linalg;
pub use sparse::prod;
pub use sparse::binop;

pub mod vec {
    pub use sparse::{
        CsVecBase,
        CsVec,
        CsVecView,
        CsVecViewMut,
    };

    pub use sparse::vec::{
        NnzIndex,
        VecDim,
        VectorIterator,
        VectorIteratorMut,
        SparseIterTools,
        IntoSparseVecIter,
        NnzOrZip,
        NnzEither,
    };
}

pub use sparse::construct::{
    vstack,
    hstack,
    bmat,
    csr_from_dense,
    csc_from_dense,
};

pub use sparse::to_dense::{
    assign_to_dense,
};



/// The shape of a matrix. This a 2-tuple with the first element indicating
/// the number of rows, and the second element indicating the number of
/// columns.
pub type Shape = (usize, usize); // FIXME: maybe we could use Ix2 here?


pub type SpRes<T> = Result<T, errors::SprsError>;

#[cfg(test)]
mod test_data;
