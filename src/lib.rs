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
let eye : CsMatOwned<f64> = CsMat::eye(sprs::CSR, 3);
let a = CsMat::new_owned(sprs::CSC, 3, 3,
                         vec![0, 2, 4, 5],
                         vec![0, 1, 0, 2, 2],
                         vec![1., 2., 3., 4., 5.]).unwrap();
```

Matrix vector multiplication

```rust
use sprs::{CsMat, CsVec};
let eye = CsMat::eye(sprs::CSR, 5);
let x = CsVec::new_owned(5, vec![0, 2, 4], vec![1., 2., 3.]).unwrap();
let y = &eye * &x;
assert_eq!(x, y);
```

Matrix matrix multiplication, addition

```rust
use sprs::{CsMat, CsVec};
let eye = CsMat::eye(sprs::CSR, 3);
let a = CsMat::new_owned(sprs::CSC, 3, 3,
                         vec![0, 2, 4, 5],
                         vec![0, 1, 0, 2, 2],
                         vec![1., 2., 3., 4., 5.]).unwrap();
let b = &eye * &a;
assert_eq!(a, b.to_csc());
```

*/

extern crate num;
extern crate ndarray;

pub mod sparse;
pub mod errors;
pub mod stack;

pub use sparse::{CsMat, CsMatOwned, CsMatView,
                 CsVec, CsVecView, CsVecOwned};
pub use sparse::CompressedStorage::{CSR, CSC};
pub use sparse::construct::{vstack, hstack, bmat};

mod utils {
    use sparse::csmat::{self, CsMatView};

    /// Create a borrowed CsMat matrix from sliced data without
    /// checking validity. Intended for internal use only.
    pub fn csmat_borrowed_uchk<'a, N>(storage: csmat::CompressedStorage,
                                      nrows : usize, ncols: usize,
                                      indptr : &'a [usize],
                                      indices : &'a [usize],
                                      data : &'a [N]
                                     ) -> CsMatView<'a, N> {
        // not actually memory unsafe here since data comes from slices
        unsafe {
            CsMatView::new_raw(storage, nrows, ncols,
                               indptr.as_ptr(),
                               indices.as_ptr(),
                               data.as_ptr())
        }
    }
}

#[cfg(test)]
mod test_data;
