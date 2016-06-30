/*!
# sprs

sprs is a sparse linear algebra library for Rust.

It features a sparse matrix type, CsMat, and a sparse vector type, CsVec,
both based on the compressed storage scheme.

All matrix algebra operations are supported, and support for direct sparse
solvers is planned.

## Examples

Matrix construction

```rust
use sprs::{CsMat, CsMatOwned, CsVec};
let eye : CsMatOwned<f64> = CsMat::eye(3);
let a = CsMat::new_csc((3, 3),
                       vec![0, 2, 4, 5],
                       vec![0, 1, 0, 2, 2],
                       vec![1., 2., 3., 4., 5.]);
```

Matrix vector multiplication

```rust
use sprs::{CsMat, CsVec};
let eye = CsMat::eye(5);
let x = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
let y = &eye * &x;
assert_eq!(x, y);
```

Matrix matrix multiplication, addition

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

pub use ndarray::Ix as Ix_;

pub use sparse::{
    CsMat,
    CsMatOwned,
    CsMatView,
    CsVec,
    CsVecView,
    CsVecOwned,
};


pub use sparse::symmetric::{
    is_symmetric,
};

pub use sparse::permutation::{
    Permutation,
    PermView,
    PermOwned,
};

pub use sparse::CompressedStorage::{
    self,
    CSR,
    CSC,
};

pub use sparse::linalg;
pub use sparse::prod;
pub use sparse::binop;
pub use sparse::vec;

pub use sparse::triplet::{
    TripletMat,
    TripletMatView,
    TripletMatViewMut,
};

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


pub type Ix2 = (Ix_, Ix_);


/// The shape of a matrix. This a 2-tuple with the first element indicating
/// the number of rows, and the second element indicating the number of
/// columns.
pub type Shape = (usize, usize); // FIXME: maybe we could use Ix2 here?


pub type SpRes<T> = Result<T, errors::SprsError>;



mod utils {

    use sparse::{csmat, CsMatView};
    use ::Shape;

    /// Create a borrowed CsMat matrix from sliced data without
    /// checking validity. Intended for internal use only.
    pub fn csmat_borrowed_uchk<'a, N>(storage: csmat::CompressedStorage,
                                      shape: Shape,
                                      indptr : &'a [usize],
                                      indices : &'a [usize],
                                      data : &'a [N]
                                     ) -> CsMatView<'a, N> {
        // not actually memory unsafe here since data comes from slices
        unsafe {
            CsMatView::new_view_raw(storage, shape,
                                    indptr.as_ptr(),
                                    indices.as_ptr(),
                                    data.as_ptr())
        }
    }
}

#[cfg(test)]
mod test_data;
