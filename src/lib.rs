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
                         vec![0, 2, 4],
                         vec![0, 1, 0, 2, 2],
                         vec![1., 2., 3., 4., 5.]).unwrap();
let b = &eye * &a;
assert_eq!(a, b);
```

*/

extern crate num;

pub mod sparse;
pub mod errors;

pub use sparse::{CsMat, CsMatOwned, CsMatView};
pub use sparse::vec::{CsVec, CsVecView, CsVecOwned};
pub use sparse::CompressedStorage::{CSR, CSC};
pub use sparse::construct::{vstack, hstack, bmat};

#[cfg(test)]
mod test_data;
