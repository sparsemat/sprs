///! Sparse matrix product

use std::ops::{Add, Mul};
use sparse::csmat::CompressedStorage::{CSC, CSR};
use sparse::csmat::{BorrowedCsMat};

pub fn mul_acc_mat_vec_csc<N: Add<Output=N> + Mul<Output=N> + Clone + Copy>(
    theMat: BorrowedCsMat<N>, inVec: &[N], resVec: &mut[N]) {
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
