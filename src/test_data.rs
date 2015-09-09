//! Some matrices used in tests

use sparse::{CsMat, CsMatOwned};
use sparse::CompressedStorage::{CSR, CSC};

pub fn mat1() -> CsMatOwned<f64> {
    let indptr = vec![0, 2, 4, 5, 6, 7];
    let indices = vec![2, 3, 3, 4, 2, 1, 3];
    let data = vec![3., 4., 2., 5., 5., 8., 7.];
    CsMat::new_owned(CSR, 5, 5, indptr, indices, data).unwrap()
}

pub fn mat1_csc() -> CsMatOwned<f64> {
    let indptr = vec![0, 0, 1, 3, 6, 7];
    let indices = vec![3, 0, 2, 0, 1, 4, 1];
    let data = vec![8.,  3.,  5.,  4.,  2.,  7.,  5.];
    CsMat::new_owned(CSC, 5, 5, indptr, indices, data).unwrap()
}

pub fn mat2() -> CsMatOwned<f64> {
    let indptr = vec![0,  4,  6,  6,  8, 10];
    let indices = vec![0, 1, 2, 4, 0, 3, 2, 3, 1, 2];
    let data = vec![6.,  7.,  3.,  3.,  8., 9.,  2.,  4.,  4.,  4.];
    CsMat::new_owned(CSR, 5, 5, indptr, indices, data).unwrap()
}

pub fn mat3() -> CsMatOwned<f64> {
    let indptr = vec![0, 2, 4, 5, 6, 7];
    let indices = vec![2, 3, 2, 3, 2, 1, 3];
    let data = vec![3., 4., 2., 5., 5., 8., 7.];
    CsMat::new_owned(CSR, 5, 4, indptr, indices, data).unwrap()
}

pub fn mat4() -> CsMatOwned<f64> {
    let indptr = vec![0,  4,  6,  6,  8, 10];
    let indices = vec![0, 1, 2, 4, 0, 3, 2, 3, 1, 2];
    let data = vec![6.,  7.,  3.,  3.,  8., 9.,  2.,  4.,  4.,  4.];
    CsMat::new_owned(CSC, 5, 5, indptr, indices, data).unwrap()
}

/// Returns the scalar product of mat1 and mat2
pub fn mat1_times_2() -> CsMatOwned<f64> {
    let indptr = vec![0, 2, 4, 5, 6, 7];
    let indices = vec![2, 3, 3, 4, 2, 1, 3];
    let data = vec![6., 8., 4., 10., 10., 16., 14.];
    CsMat::new_owned(CSR, 5, 5, indptr, indices, data).unwrap()
}

// Matrix product of mat1 with itself
pub fn mat1_self_matprod() -> CsMatOwned<f64> {
    let indptr = vec![0, 2, 4, 5, 7, 8];
    let indices = vec![1, 2, 1, 3, 2, 3, 4, 1];
    let data = vec![32., 15., 16., 35., 25., 16., 40., 56.];
    CsMat::new_owned(CSR, 5, 5, indptr, indices, data).unwrap()
}

pub fn mat1_matprod_mat2() -> CsMatOwned<f64> {
    let indptr = vec![0, 2, 5, 5, 7, 9];
    let indices = vec![2, 3, 1, 2, 3, 0, 3, 2, 3];
    let data = vec![8., 16., 20., 24.,  8., 64., 72., 14., 28.];
    CsMat::new_owned(CSR, 5, 5, indptr, indices, data).unwrap()
}

pub fn mat1_csc_matprod_mat4() -> CsMatOwned<f64> {
    let indptr = vec![0,  4,  7,  7, 11, 14];
    let indices = vec![0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 4, 0, 2, 3];
    let data = vec![9., 15., 15., 56., 36., 18., 63., 22.,
                    8., 10., 28., 12., 20., 32.];
    CsMat::new_owned(CSC, 5, 5, indptr, indices, data).unwrap()
}
