//! Some matrices used in tests

use sparse::{CsMat};
use sparse::CompressedStorage::{CSR, CSC};

pub fn mat1() -> CsMat<f64, Vec<usize>, Vec<f64>> {
    let indptr = vec![0, 2, 4, 5, 6, 7];
    let indices = vec![2, 3, 3, 4, 2, 1, 3];
    let data = vec![3., 4., 2., 5., 5., 8., 7.];
    CsMat::from_vecs(CSR, 5, 5, indptr, indices, data).unwrap()
}

pub fn mat1_csc() -> CsMat<f64, Vec<usize>, Vec<f64>> {
    let indptr = vec![0, 0, 1, 3, 6, 7];
    let indices = vec![3, 0, 2, 0, 1, 4, 1];
    let data = vec![8.,  3.,  5.,  4.,  2.,  7.,  5.];
    CsMat::from_vecs(CSC, 5, 5, indptr, indices, data).unwrap()
}

pub fn mat2() -> CsMat<f64, Vec<usize>, Vec<f64>> {
    let indptr = vec![0,  4,  6,  6,  8, 10];
    let indices = vec![0, 1, 2, 4, 0, 3, 2, 3, 1, 2];
    let data = vec![6.,  7.,  3.,  3.,  8., 9.,  2.,  4.,  4.,  4.];
    CsMat::from_vecs(CSR, 5, 5, indptr, indices, data).unwrap()
}

pub fn mat3() -> CsMat<f64, Vec<usize>, Vec<f64>> {
    let indptr = vec![0, 2, 4, 5, 6, 7];
    let indices = vec![2, 3, 2, 3, 2, 1, 3];
    let data = vec![3., 4., 2., 5., 5., 8., 7.];
    CsMat::from_vecs(CSR, 5, 4, indptr, indices, data).unwrap()
}

pub fn mat4() -> CsMat<f64, Vec<usize>, Vec<f64>> {
    let indptr = vec![0,  4,  6,  6,  8, 10];
    let indices = vec![0, 1, 2, 4, 0, 3, 2, 3, 1, 2];
    let data = vec![6.,  7.,  3.,  3.,  8., 9.,  2.,  4.,  4.,  4.];
    CsMat::from_vecs(CSC, 5, 5, indptr, indices, data).unwrap()
}

