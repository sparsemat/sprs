//! Implementation of the paper
//! Bank and Douglas, 2001, Sparse Matrix Multiplication Package (SMPP)

use indexing::SpIndex;

/// Compute the symbolic structure of the
/// matrix product C = A * B
///
/// `index.len()` should be equal to the maximum dimension among the input
/// matrices.
///
/// This algorithm has a complexity of O(n * k * log(k)), where k is the
/// average number of nonzeros in the rows of the result.
pub fn symbolic<Iptr: SpIndex, I: SpIndex>(
    a_indptr: &[Iptr],
    a_indices: &[I],
    b_cols: usize,
    b_indptr: &[Iptr],
    b_indices: &[I],
    c_indptr: &mut [Iptr],
    // TODO look for litterature on the nnz of C to be able to have a slice here
    c_indices: &mut Vec<I>,
    index: &mut [usize],
) {
    assert!(a_indptr.len() == c_indptr.len());
    let a_rows = a_indptr.len() - 1;
    let b_rows = b_indptr.len() - 1;
    let a_nnz = a_indptr[a_rows].index();
    let b_nnz = b_indptr[b_rows].index();
    c_indices.reserve_exact(a_nnz + b_nnz);

    // `index` is used as a set to remember which columns of a row of C are
    // nonzero. At any point in the algorithm, if `index[col] == sentinel0`,
    // then we know there is no nonzero value in the column. As the algorithm
    // progresses, we discover nonzero elements. When a nonzero at `col` is
    // discovered, we store in `index[col]` the column of the preceding
    // nonzero (storing `sentinel1` for the first nonzero). Therefore,
    // when we want to collect nonzeros and clear the set, we can simply
    // follow the trail of column indices, putting back `sentinel0` along
    // the way. This way, collecting the nonzero indices for a column
    // has a complexity O(col_nnz).
    let ind_len = a_rows.max(b_rows.max(b_cols));
    let sentinel0 = usize::max_value();
    let sentinel1 = usize::max_value() - 1;
    assert!(index.len() == ind_len);
    assert!(ind_len < sentinel1);
    for elt in index.iter_mut() {
        *elt = sentinel0;
    }

    c_indptr[0] = Iptr::from_usize(0);
    for a_row in 0..a_rows {
        let mut istart = sentinel1;
        let mut length = 0;
        let a_start = a_indptr[a_row].index();
        let a_stop = a_indptr[a_row + 1].index();
        for &a_col in &a_indices[a_start..a_stop] {
            let b_row = a_col.index();
            let b_start = b_indptr[b_row].index();
            let b_stop = b_indptr[b_row + 1].index();
            for b_col in &b_indices[b_start..b_stop] {
                let b_col = b_col.index();
                if index[b_col] == sentinel0 {
                    index[b_col] = istart;
                    istart = b_col;
                    length += 1;
                }
            }
        }
        c_indptr[a_row + 1] = c_indptr[a_row] + Iptr::from_usize(length);
        for _ in 0..length {
            debug_assert!(istart < sentinel1);
            c_indices.push(I::from_usize(istart));
            let new_start = index[istart];
            index[istart] = sentinel0;
            istart = new_start;
        }
        let c_start = c_indptr[a_row].index();
        let c_end = c_indptr[a_row + 1].index();
        c_indices[c_start..c_end].sort();
        index[a_row] = sentinel0;
    }
}

#[cfg(test)]
mod test {
    use test_data;

    #[test]
    fn symbolic() {
        let a = test_data::mat1();
        let b = test_data::mat2();
        // a * b 's structure:
        //                | x x x   x |
        //                | x     x   |
        //                |           |
        //                |     x x   |
        //                |   x x     |
        //
        // |     x x   |  |     x x   |
        // |       x x |  |   x x x   |
        // |     x     |  |           |
        // |   x       |  | x     x   |
        // |       x   |  |     x x   |
        let exp = test_data::mat1_matprod_mat2();

        let mut c_indptr = [0; 6];
        let mut c_indices = Vec::new();
        let mut index = [0; 5];

        super::symbolic(
            a.indptr(),
            a.indices(),
            b.cols(),
            b.indptr(),
            b.indices(),
            &mut c_indptr,
            &mut c_indices,
            &mut index,
        );

        assert_eq!(exp.indptr(), c_indptr);
        assert_eq!(exp.indices(), &c_indices[..]);
    }
}
