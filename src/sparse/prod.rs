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


#[cfg(test)]
mod test {
    use sparse::csmat::{new_borrowed_csmat};
    use sparse::csmat::CompressedStorage::{CSC, CSR};
    use super::mul_acc_mat_vec_csc;
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
}
