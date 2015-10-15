/// Sparse triangular solves

use num::traits::Num;
use sparse::csmat;
use errors::SprsError;

fn check_solver_dimensions<N>(lower_tri_mat: &csmat::CsMatView<N>,
                              rhs: &[N]) -> Result<(), SprsError>
where N: Copy + Num {
    let (cols, rows) = (lower_tri_mat.cols(),lower_tri_mat.rows());
    if  cols != rows {
        return Err(SprsError::NonSquareMatrix);
    }
    if  cols != rhs.len() {
        return Err(SprsError::IncompatibleDimensions);
    }
    Ok(())
}

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
    try!(check_solver_dimensions(&lower_tri_mat, rhs));
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


/// Solve a sparse lower triangular matrix system, with a csc matrix
/// and a dense vector as inputs
///
/// The solve results are written into the provided values.
///
/// This method does not require the matrix to actually be lower triangular,
/// but is most efficient if the first element of each column
/// is the diagonal element (thus actual sorted lower triangular matrices work
/// best). Otherwise, logarithmic search for the diagonal element
/// has to be performed for each column.
pub fn lsolve_csc_dense_rhs<N>(lower_tri_mat: csmat::CsMatView<N>,
                               rhs: &mut [N]) -> Result<(), SprsError>
where N: Copy + Num {
    try!(check_solver_dimensions(&lower_tri_mat, rhs));
    if ! lower_tri_mat.is_csc() {
        return Err(SprsError::BadStorageType);
    }

    // we base our algorithm on the following decomposition:
    // |l_0_0    0    | |x_0|    |b_0|
    // |l_1_0    L_1_1| |x_1|  = |b_1|
    //
    // At each step of the algorithm, the x_0 part is computed as b_0 / l_0_0
    // and the step can be propagated by solving the reduced system
    // L_1_1 x1 = b_1 - x0*l_1_0

    for (col_ind, col) in lower_tri_mat.outer_iterator() {
        if let Some(diag_val) = col.at(col_ind) {
            if diag_val == N::zero() {
                return Err(SprsError::SingularMatrix);
            }
            let b = rhs[col_ind];
            let x = b / diag_val;
            rhs[col_ind] = x;
            for (row_ind, val) in col.iter() {
                if row_ind <= col_ind {
                    continue;
                }
                let b = rhs[row_ind];
                rhs[row_ind] = b - val * x;
            }
        }
        else {
            return Err(SprsError::SingularMatrix);
        }
    }
    Ok(())
}

/// Solve a sparse upper triangular matrix system, with a csc matrix
/// and a dense vector as inputs
///
/// The solve results are written into the provided values.
///
/// This method does not require the matrix to actually be lower triangular,
/// but is most efficient if the last element of each column
/// is the diagonal element (thus actual sorted lower triangular matrices work
/// best). Otherwise, logarithmic search for the diagonal element
/// has to be performed for each column.
pub fn usolve_csc_dense_rhs<N>(upper_tri_mat: csmat::CsMatView<N>,
                               rhs: &mut [N]) -> Result<(), SprsError>
where N: Copy + Num {
    try!(check_solver_dimensions(&upper_tri_mat, rhs));
    if ! upper_tri_mat.is_csc() {
        return Err(SprsError::BadStorageType);
    }
    //
    // we base our algorithm on the following decomposition:
    // | U_0_0    u_0_1 | | x_0 |    | b_0 |
    // |   0      u_1_1 | | x_1 |  = | b_1 |
    //
    // At each step of the algorithm, the x_1 part is computed as b_1 / u_1_1
    // and the step can be propagated by solving the reduced system
    // U_0_0 x0 = b_0 - x1*u_0_1

    for (col_ind, col) in upper_tri_mat.outer_iterator().rev() {
        if let Some(diag_val) = col.at(col_ind) {
            if diag_val == N::zero() {
                return Err(SprsError::SingularMatrix);
            }
            let b = rhs[col_ind];
            let x = b / diag_val;
            rhs[col_ind] = x;
            for (row_ind, val) in col.iter() {
                if row_ind >= col_ind {
                    continue;
                }
                let b = rhs[row_ind];
                rhs[row_ind] = b - val * x;
            }
        }
        else {
            return Err(SprsError::SingularMatrix);
        }
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

    #[test]
    fn lsolve_csc_dense_rhs() {
        // |1    | |3|   |3|
        // |1 2  | |1| = |5|
        // |0 0 3| |1|   |3|
        let l = csmat::CsMatOwned::new_owned(csmat::CompressedStorage::CSC,
                                             3, 3, vec![0, 2, 3, 4],
                                             vec![0, 1, 1, 2],
                                             vec![1, 1, 2, 3]).unwrap();
        let b = vec![3, 5, 3];
        let mut x = b.clone();

        super::lsolve_csc_dense_rhs(l.borrowed(), &mut x).unwrap();
        assert_eq!(x, vec![3, 1, 1]);
    }

    #[test]
    fn usolve_csc_dense_rhs() {
        // |1 0 1| |3|   |4|
        // |  2 0| |1| = |2|
        // |    3| |1|   |3|
        let u = csmat::CsMatOwned::new_owned(csmat::CompressedStorage::CSC,
                                             3, 3, vec![0, 1, 2, 4],
                                             vec![0, 1, 0, 2],
                                             vec![1, 2, 1, 3]).unwrap();
        let b = vec![4, 2, 3];
        let mut x = b.clone();

        super::usolve_csc_dense_rhs(u.borrowed(), &mut x).unwrap();
        assert_eq!(x, vec![3, 1, 1]);
    }
}
