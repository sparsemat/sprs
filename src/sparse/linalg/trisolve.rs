/// Sparse triangular solves

use num::traits::Num;
use sparse::csmat;
use errors::SprsError;

/// Solve a sparse lower triangular matrix system, with a csr matrix
/// and a dense vector as inputs
/// 
/// The solve results are written into the provided values.
///
/// This solve does not assume the input matrix to actually be
/// triangular, instead it ignores the upper triangular part.
pub fn lsolve_csr_dense_rhs<N>(lower_tri_mat: csmat::CsMatView<N>,
                               rhs: &mut [N]) -> Result<(), SprsError>
where N: Copy + Num {
    if ! lower_tri_mat.is_csr() {
        return Err(SprsError::BadStorageType);
    }

    // we base our algorithm on the following decomposition:
    // | L_0_0    0     | | x_0 |    | b_0 |
    // | l_1_0^T  l_1_1 | | x_1 |  = | b_1 |
    // 
    // At each step of the algorithm, the x_0 part is known,
    // and x_1 can be computed as x_1 = (b_1 - l_1_0^T.b_0) / l_1_1

    for (row_ind, row) in lower_tri_mat.outer_iterator() {
        let mut diag_val = N::zero();
        let mut x = rhs[row_ind];
        for (col_ind, val) in row.iter() {
            if col_ind == row_ind {
                diag_val = val;
                continue;
            }
            if col_ind > row_ind {
                continue;
            }
            x = x - val * rhs[col_ind];
        }
        if diag_val == N::zero() {
            return Err(SprsError::SingularMatrix);
        }
        rhs[row_ind] = x / diag_val;
    }
    Ok(())
}


#[cfg(test)]
mod test {

    use sparse::csmat;

    #[test]
    fn lsolve_csr_dense_rhs() {
        // |1    | |3|   |3|
        // |0 2  | |1| = |2|
        // |1 0 1| |1|   |4|
        let l = csmat::CsMatOwned::new_owned(csmat::CompressedStorage::CSR,
                                             3, 3, vec![0, 1, 2, 4],
                                             vec![0, 1, 0, 2],
                                             vec![1, 2, 1, 1]).unwrap();
        let b = vec![3, 2, 4];
        let mut x = b.clone();

        super::lsolve_csr_dense_rhs(l.borrowed(), &mut x).unwrap();
        assert_eq!(x, vec![3, 1, 1]);
    }
}
