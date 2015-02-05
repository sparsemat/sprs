/*!
CSRust
======

CSRust is a sparse linear algebra library for Rust.

*/

#![feature(core)]

extern crate num;

pub use sparse::{
    CompressedStorage,
    CsMat,
    mul_acc_mat_vec_csc,
};

mod sparse;
mod dense;

