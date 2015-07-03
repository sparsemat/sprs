///! Sparse matrix addition, subtraction

use std::ops::{Deref};
use sparse::csmat::{CsMat};
use num::traits::Num;
use sparse::vec::NnzEither::{Left, Right, Both};
use sparse::vec::CsVec;

pub fn add_mat_same_storage<N, IStorage, DStorage>(
    lhs: &CsMat<N, IStorage, DStorage>,
    rhs: &CsMat<N, IStorage, DStorage>) -> CsMat<N, Vec<usize>, Vec<N>>
where
N: Num + Copy,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {
    // TODO: return a Result<CsMat, SprsError> ?
    assert_eq!(lhs.cols(), rhs.cols());
    assert_eq!(lhs.rows(), rhs.rows());
    assert_eq!(lhs.storage_type(), rhs.storage_type());

    // TODO: do we want to expose the workspace?
    let mut res = CsMat::empty(lhs.storage_type(), lhs.inner_dims());
    res.reserve_nnz_exact(lhs.nb_nonzero() + rhs.nb_nonzero());
    res.reserve_outer_dim_exact(lhs.outer_dims());
    let mut bufvec = CsVec::empty(lhs.inner_dims());
    bufvec.reserve(lhs.inner_dims());

    for ((_, lv), (_, rv)) in lhs.outer_iterator().zip(rhs.outer_iterator()) {
        for elem in lv.iter().nnz_or_zip(rv.iter()) {
            match elem {
                Left((ind, val)) => bufvec.append(ind, val),
                Right((ind, val)) => bufvec.append(ind, val),
                Both((ind, lval, rval)) => bufvec.append(ind, lval + rval)
            }
        }
        res = res.append_outer_csvec(bufvec.borrowed());
        bufvec.clear();
    }
    assert_eq!(lhs.rows(), res.rows());
    res
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
