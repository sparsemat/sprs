///! Sparse matrix addition, subtraction

use std::ops::{Deref};
use sparse::csmat::{CsMat};
use num::traits::Num;
use sparse::vec::NnzEither::{Left, Right, Both};

/// Sparse matrix addition, with matrices sharing the same storage type
pub fn add_mat_same_storage<N, IStorage, DStorage>(
    lhs: &CsMat<N, IStorage, DStorage>,
    rhs: &CsMat<N, IStorage, DStorage>) -> CsMat<N, Vec<usize>, Vec<N>>
where
N: Num + Copy,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {
    let binop = |x, y| x + y;
    return csmat_binop_same_storage_alloc(lhs.borrowed(), rhs.borrowed(),
                                          binop);
}

/// Sparse matrix subtraction, with same storage type
pub fn sub_mat_same_storage<N, IStorage, DStorage>(
    lhs: &CsMat<N, IStorage, DStorage>,
    rhs: &CsMat<N, IStorage, DStorage>) -> CsMat<N, Vec<usize>, Vec<N>>
where
N: Num + Copy,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {
    let binop = |x, y| x - y;
    return csmat_binop_same_storage_alloc(lhs.borrowed(), rhs.borrowed(),
                                          binop);
}

/// Sparse matrix scalar multiplication, with same storage type
pub fn mul_mat_same_storage<N, IStorage, DStorage>(
    lhs: &CsMat<N, IStorage, DStorage>,
    rhs: &CsMat<N, IStorage, DStorage>) -> CsMat<N, Vec<usize>, Vec<N>>
where
N: Num + Copy,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {
    let binop = |x, y| x * y;
    return csmat_binop_same_storage_alloc(lhs.borrowed(), rhs.borrowed(),
                                          binop);
}

/// Sparse matrix scalar division, with same storage type
pub fn div_mat_same_storage<N, IStorage, DStorage>(
    lhs: &CsMat<N, IStorage, DStorage>,
    rhs: &CsMat<N, IStorage, DStorage>) -> CsMat<N, Vec<usize>, Vec<N>>
where
N: Num + Copy,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {
    let binop = |x, y| x / y;
    return csmat_binop_same_storage_alloc(lhs.borrowed(), rhs.borrowed(),
                                          binop);
}


fn csmat_binop_same_storage_alloc<N, F>(
    lhs: CsMat<N, &[usize], &[N]>,
    rhs: CsMat<N, &[usize], &[N]>,
    binop: F) -> CsMat<N, Vec<usize>, Vec<N>>
where
N: Num + Copy,
F: Fn(N, N) -> N {
    // TODO: return a Result<CsMat, SprsError> ?
    let nrows = lhs.rows();
    let ncols = lhs.cols();
    let storage_type = lhs.storage_type();
    assert_eq!(nrows, rhs.cols());
    assert_eq!(ncols, rhs.rows());
    assert_eq!(storage_type, rhs.storage_type());

    let max_nnz = lhs.nb_nonzero() + rhs.nb_nonzero();
    let mut out_indptr = vec![0; lhs.outer_dims() + 1];
    let mut out_indices = vec![0; max_nnz];
    let mut out_data = vec![N::zero(); max_nnz];
    let nnz = csmat_binop_same_storage_raw(lhs, rhs, binop,
                                           &mut out_indptr[..],
                                           &mut out_indices[..],
                                           &mut out_data[..]);
    out_indices.truncate(nnz);
    out_data.truncate(nnz);
    CsMat::from_vecs(storage_type, nrows, ncols,
                     out_indptr, out_indices, out_data).unwrap()
}

/// Raw implementation of scalar binary operation for compressed sparse matrices
/// sharing the same storage. The output arrays are assumed to be preallocated
///
/// Returns the nnz count
pub fn csmat_binop_same_storage_raw<N, F>(
    lhs: CsMat<N, &[usize], &[N]>,
    rhs: CsMat<N, &[usize], &[N]>,
    binop: F,
    out_indptr: &mut [usize],
    out_indices: &mut [usize],
    out_data: &mut [N]
    ) -> usize
where
N: Num + Copy,
F: Fn(N, N) -> N {
    assert_eq!(lhs.cols(), rhs.cols());
    assert_eq!(lhs.rows(), rhs.rows());
    assert_eq!(lhs.storage_type(), rhs.storage_type());
    assert_eq!(out_indptr.len(), rhs.outer_dims() + 1);
    let max_nnz = lhs.nb_nonzero() + rhs.nb_nonzero();
    assert!(out_data.len() >= max_nnz);
    assert!(out_indices.len() >= max_nnz);
    let mut nnz = 0;
    out_indptr[0] = 0;
    for ((dim, lv), (_, rv)) in lhs.outer_iterator().zip(rhs.outer_iterator()) {
        for elem in lv.iter().nnz_or_zip(rv.iter()) {
            let (ind, binop_val) = match elem {
                Left((ind, val)) => (ind, binop(val, N::zero())),
                Right((ind, val)) => (ind, binop(N::zero(), val)),
                Both((ind, lval, rval)) => (ind, binop(lval, rval)),
            };
            if binop_val != N::zero() {
                out_indices[nnz] = ind;
                out_data[nnz] = binop_val;
                nnz += 1;
            }
        }
        out_indptr[dim+1] = nnz;
    }
    nnz
}

#[cfg(test)]
mod test {
    use sparse::csmat::CsMat;
    use super::add_mat_same_storage;
    use sparse::csmat::CompressedStorage::{CSC, CSR};

    fn mat1() -> CsMat<f64, Vec<usize>, Vec<f64>> {
        let indptr = vec![0, 2, 4, 5, 6, 7];
        let indices = vec![2, 3, 3, 4, 2, 1, 3];
        let data = vec![3., 4., 2., 5., 5., 8., 7.];
        CsMat::from_vecs(CSR, 5, 5, indptr, indices, data).unwrap()
    }

    fn mat2() -> CsMat<f64, Vec<usize>, Vec<f64>> {
        let indptr = vec![0,  4,  6,  6,  8, 10];
        let indices = vec![0, 1, 2, 4, 0, 3, 2, 3, 1, 2];
        let data = vec![6.,  7.,  3.,  3.,  8., 9.,  2.,  4.,  4.,  4.];
        CsMat::from_vecs(CSR, 5, 5, indptr, indices, data).unwrap()
    }

    fn mat1_plus_mat2() -> CsMat<f64, Vec<usize>, Vec<f64>> {
        let indptr = vec![0,  5,  8,  9, 12, 15];
        let indices = vec![0, 1, 2, 3, 4, 0, 3, 4, 2, 1, 2, 3, 1, 2, 3];
        let data = vec![
            6.,  7.,  6.,  4.,  3.,
            8.,  11.,  5.,  5.,  8.,
            2.,  4.,  4.,  4.,  7.];
        CsMat::from_vecs(CSR, 5, 5, indptr, indices, data).unwrap()
    }

    #[test]
    fn test_add1() {
        let a = mat1();
        let b = mat2();

        let c = add_mat_same_storage(&a, &b);
        let c_true = mat1_plus_mat2();
        assert_eq!(c.indptr(), c_true.indptr());
        assert_eq!(c.indices(), c_true.indices());
        assert_eq!(c.data(), c_true.data());
    }
}
