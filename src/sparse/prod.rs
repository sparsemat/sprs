///! Sparse matrix product

use std::ops::{Add, Mul};
use sparse::csmat::CompressedStorage::{CSC, CSR};
use sparse::csmat::{CsMat};
use num::traits::Num;

pub fn mul_acc_mat_vec_csc<N: Num + Clone + Copy>(
    theMat: CsMat<N>, inVec: &[N], resVec: &mut[N]) {
    assert!(theMat.cols() == inVec.len(), "Matrix and vector dims must agree");
    assert!(theMat.rows() == resVec.len(), "Matrix and res vector dims must agree");
    assert!(theMat.storage_type() == CSC, "Matrix must be in CSC format");

    for (col_ind, row_inds, values) in theMat.outer_iterator() {
        let multiplier = &inVec[col_ind];
        for (row_ind, value) in row_inds.iter().zip(values.iter()) {
            // TODO: unsafe access to value? needs bench
            resVec[*row_ind] =
                resVec[*row_ind] + *multiplier * *value;
        }
    }
}

pub fn mul_acc_mat_vec_csr<N: Num + Clone + Copy>(
    theMat: CsMat<N>, inVec: &[N], resVec: &mut[N]) {
    assert!(theMat.cols() == inVec.len(), "Matrix and vector dims must agree");
    assert!(theMat.rows() == resVec.len(), "Matrix and res vector dims must agree");
    assert!(theMat.storage_type() == CSR, "Matrix must be in CSR format");

    for (row_ind, col_inds, values) in theMat.outer_iterator() {
        for (col_ind, value) in col_inds.iter().zip(values.iter()) {
            // TODO: unsafe access to value? needs bench
            resVec[row_ind] =
                resVec[row_ind] + inVec[*col_ind] * *value;
        }
    }
}


#[cfg(test)]
mod test {
    use sparse::csmat::{new_borrowed_csmat};
    use sparse::csmat::CompressedStorage::{CSC, CSR};
    use super::{mul_acc_mat_vec_csc, mul_acc_mat_vec_csr};
    use std::num::Float;

    #[test]
    fn mul_csc_vec() {
        let indptr: &[usize] = &[0, 2, 4, 5, 6, 7];
        let indices: &[usize] = &[2, 3, 3, 4, 2, 1, 3];
        let data: &[f64] = &[
            0.35310881, 0.42380633, 0.28035896, 0.58082095,
            0.53350123, 0.88132896, 0.72527863];

        let mat = new_borrowed_csmat(CSC, 5, 5, indptr, indices, data).unwrap();
        let vector = vec![0.1, 0.2, -0.1, 0.3, 0.9];
        let mut resVec = vec![0., 0., 0., 0., 0.];
        mul_acc_mat_vec_csc(mat, vector.as_slice(), resVec.as_mut_slice());

        let expected_output =
            vec![ 0., 0.26439869, -0.01803924, 0.75120319, 0.11616419];

        let epsilon = 1e-7; // TODO: get better values and increase precision

        assert!(resVec.iter().zip(expected_output.iter()).all(
            |(x,y)| Float::abs(*x-*y) < epsilon));
    }

    #[test]
    fn mul_csr_vec() {
        let indptr: &[usize] = &[0, 3, 3, 5, 6, 7];
        let indices: &[usize] = &[1, 2, 3, 2, 3, 4, 4];
        let data: &[f64] = &[
            0.75672424, 0.1649078, 0.30140296, 0.10358244,
            0.6283315, 0.39244208, 0.57202407];

        let mat = new_borrowed_csmat(CSR, 5, 5, indptr, indices, data).unwrap();
        let vector = vec![0.1, 0.2, -0.1, 0.3, 0.9];
        let mut resVec = vec![0., 0., 0., 0., 0.];
        mul_acc_mat_vec_csr(mat, vector.as_slice(), resVec.as_mut_slice());

        let expected_output =
            vec![0.22527496, 0., 0.17814121, 0.35319787, 0.51482166];

        let epsilon = 1e-7; // TODO: get better values and increase precision

        assert!(resVec.iter().zip(expected_output.iter()).all(
            |(x,y)| Float::abs(*x-*y) < epsilon));
    }
}
