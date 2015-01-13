/*!
CSRust
======

CSRust is a sparse linear algebra library for Rust.

*/

pub use sparse::{
    CompressedStorage,
    CsMat,
    BorrowedCsMat,
    check_csmat_structure,
    mul_acc_mat_vec_csc,
};


mod sparse;

