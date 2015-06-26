///! Sparse matrix addition, subtraction

use std::ops::{Deref};
use sparse::csmat::CompressedStorage::{CSC, CSR};
use sparse::csmat::{CsMat};
use num::traits::Num;

pub fn add_mat_same_storage<N, IStorage, DStorage>(
    lhs: CsMat<N, IStorage, DStorage>,
    rhs: &CsMat<N, IStorage, DStorage>) -> CsMat<N, IStorage, DStorage>
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
    res.reserve_nnz_exact(lhs.nnz() + rhs.nnz());
    res.reserve_outer_dim_exact(lhs.outer_dims());
    let mut bufvec = CsVec::empty(lhs.inner_dims());
    bufvec.reserve(lhs.inner_dims());

    use spasre::vec::NnzEither::*;
    for ((_, lv), (_, rv)) in lhs.outer_iterator().zip(rhs.outer_iterator()) {
        for elem in lv.iter().nnz_or_zip(rv.iter()) {
            match elem {
                Left((ind, val)) => bufvec.append(ind, val),
                Right((ind, val)) => bufvec.append(ind, val),
                Both((ind, lval, rval)) => bufvec.append(ind, lval + rval)
            }
        }
        res = res.append_outer_csvec(bufvec);
    }
    assert_eq!(lhs.rows(), res.rows());
    res
}
