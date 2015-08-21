//! High level construction of sparse matrices by stacking, by block, ...

use std::ops::{Deref};
use sparse::csmat::{CsMatVec, CsMatView, CompressedStorage};
use errors::SprsError;

/// Stack the given matrices into a new one, using the most efficient stacking
/// direction (ie vertical stack for CSR matrices, horizontal stack for CSC)
pub fn same_storage_fast_stack<N>(
    mats: &[CsMatView<N>]) -> Result<CsMatVec<N>, SprsError>
where N: Copy {
    if mats.len() == 0 {
        return Err(SprsError::EmptyStackingList);
    }
    let inner_dim = mats[0].inner_dims();
    if ! mats.iter().all(|x| x.inner_dims() == inner_dim) {
        return Err(SprsError::IncompatibleDimensions);
    }
    let storage_type = mats[0].storage();
    if ! mats.iter().all(|x| x.storage() == storage_type) {
        return Err(SprsError::IncompatibleStorages);
    }

    let outer_dim = mats.iter().map(|x| x.outer_dims()).fold(0, |x, y| x + y);
    let nnz = mats.iter().map(|x| x.nb_nonzero()).fold(0, |x, y| x + y);

    let mut res = CsMatVec::empty(storage_type, inner_dim);
    res.reserve_outer_dim_exact(outer_dim);
    res.reserve_nnz_exact(nnz);
    for mat in mats {
        for (_, vec) in mat.outer_iterator() {
            res = res.append_outer_csvec(vec.borrowed());
        }
    }

    Ok(res)
}




#[cfg(test)]
mod test {
    use sparse::csmat::CsMat;
    use sparse::CompressedStorage::{CSR};
    use test_data::{mat1, mat2, mat3, mat4};
    use errors::SprsError::*;

    fn mat1_vstack_mat2() -> CsMat<f64, Vec<usize>, Vec<f64>> {
        let indptr = vec![0, 2, 4, 5, 6, 7, 11, 13, 13, 15, 17];
        let indices = vec![2, 3, 3, 4, 2, 1, 3, 0, 1, 2, 4, 0, 3, 2, 3, 1, 2];
        let data = vec![3., 4., 2., 5., 5., 8., 7., 6., 7., 3., 3.,
                        8., 9., 2., 4., 4., 4.];
        CsMat::from_vecs(CSR, 10, 5, indptr, indices, data).unwrap()
    }

    #[test]
    fn same_storage_fast_stack_failures() {
        let res: Result<CsMat<f64, _, _>, _> =
            super::same_storage_fast_stack(&[]);
        assert_eq!(res, Err(EmptyStackingList));
        let a = mat1();
        let c = mat3();
        let d = mat4();
        let res: Result<CsMat<f64, _, _>, _> =
            super::same_storage_fast_stack(&[]);
        let res = super::same_storage_fast_stack(&[a.borrowed(), c.borrowed()]);
        assert_eq!(res, Err(IncompatibleDimensions));
        let res = super::same_storage_fast_stack(&[a.borrowed(), d.borrowed()]);
        assert_eq!(res, Err(IncompatibleStorages));
    }

    #[test]
    fn same_storage_fast_stack_ok() {
        let a = mat1();
        let b = mat2();
        let res = super::same_storage_fast_stack(&[a.borrowed(), b.borrowed()]);
        let expected = mat1_vstack_mat2();
        assert_eq!(res, Ok(expected));
    }
}
