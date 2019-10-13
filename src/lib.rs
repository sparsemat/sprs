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

#[cfg(feature = "alga")]
extern crate alga;
extern crate ndarray;
extern crate num_complex;
extern crate num_traits;
#[cfg(feature = "serde")]
extern crate serde;
#[cfg(feature = "serde_derive")]
#[macro_use]
extern crate serde_derive;
#[cfg(test)]
extern crate tempdir;

pub mod array_backend;
pub mod errors;
pub mod indexing;
pub mod io;
mod num_kinds;
mod sparse;
pub mod stack;

/// Deprecated type alias, will be removed on next breaking change
pub type Ix_ = ndarray::Ix1;
pub type Ix1 = ndarray::Ix1;
pub type Ix2 = ndarray::Ix2;

pub use indexing::SpIndex;

pub use sparse::{
    CsMat, CsMatBase, CsMatI, CsMatVecView, CsMatView, CsMatViewI,
    CsMatViewMut, CsMatViewMutI, CsVec, CsVecBase, CsVecI, CsVecView,
    CsVecViewI, CsVecViewMut, CsVecViewMutI, SparseMat, TriMat, TriMatBase,
    TriMatI, TriMatIter, TriMatView, TriMatViewI, TriMatViewMut,
    TriMatViewMutI,
};

pub use sparse::symmetric::is_symmetric;

pub use sparse::permutation::{
    PermOwned, PermOwnedI, PermView, PermViewI, Permutation,
};

pub use sparse::CompressedStorage::{self, CSC, CSR};

pub use sparse::binop;
pub use sparse::linalg;
pub use sparse::prod;

pub mod vec {
    pub use sparse::{CsVec, CsVecBase, CsVecView, CsVecViewMut};

    pub use sparse::vec::{
        IntoSparseVecIter, NnzEither, NnzIndex, NnzOrZip, SparseIterTools,
        VecDim, VectorIterator, VectorIteratorMut,
    };
}

pub use sparse::construct::{bmat, hstack, vstack};

pub use sparse::to_dense::assign_to_dense;

/// The shape of a matrix. This a 2-tuple with the first element indicating
/// the number of rows, and the second element indicating the number of
/// columns.
pub type Shape = (usize, usize); // FIXME: maybe we could use Ix2 here?

pub type SpRes<T> = Result<T, errors::SprsError>;

#[cfg(test)]
mod test_data;

#[cfg(test)]
mod test {
    use super::CsMat;

    #[test]
    fn iter_rbr() {
        let mat = CsMat::new(
            (3, 3),
            vec![0, 2, 3, 3],
            vec![1, 2, 0],
            vec![0.1, 0.2, 0.3],
        );
        let view = mat.view();
        let mut iter = view.iter();
        assert_eq!(iter.next(), Some((&0.1, (0, 1))));
        assert_eq!(iter.next(), Some((&0.2, (0, 2))));
        assert_eq!(iter.next(), Some((&0.3, (1, 0))));
        assert_eq!(iter.next(), None);
    }
}
