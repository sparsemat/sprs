//! High level construction of sparse matrices by stacking, by block, ...

use std::ops::{Deref};
use sparse::csmat::{CsMat, CompressedStorage};
use errors::SprsError;

/// Stack the given matrices into a new one, using the most efficient stacking
/// direction (ie vertical stack for CSR matrices, horizontal stack for CSC)
pub fn same_storage_fast_stack<N>(
    mats: &[CsMat<N, &[usize], &[N]>]
    ) -> Result<CsMat<N, Vec<usize>, Vec<N>>, SprsError>
where N: Copy {
    if mats.len() == 0 {
        return Err(SprsError::EmptyStackingList);
    }
    let inner_dim = mats[0].inner_dims();
    if ! mats.iter().all(|x| x.inner_dims() == inner_dim) {
        return Err(SprsError::IncompatibleDimensions);
    }
    let storage_type = mats[0].storage_type();
    if ! mats.iter().all(|x| x.storage_type() == storage_type) {
        return Err(SprsError::IncompatibleStorages);
    }

    let outer_dim = mats.iter().map(|x| x.outer_dims()).fold(0, |x, y| x + y);
    let nnz = mats.iter().map(|x| x.nb_nonzero()).fold(0, |x, y| x + y);

    let mut res = CsMat::empty(storage_type, inner_dim);
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

    #[test]
    fn same_storage_fast_stack_failures() {
        let res: Result<CsMat<f64, _, _>, _> =
            super::same_storage_fast_stack(&[]);
        assert_eq!(res, Err(EmptyStackingList));
        let a = mat1();
        let b = mat2();
        let c = mat3();
        let d = mat4();
        let res: Result<CsMat<f64, _, _>, _> =
            super::same_storage_fast_stack(&[]);
        let res = super::same_storage_fast_stack(&[a.borrowed(), c.borrowed()]);
        assert_eq!(res, Err(IncompatibleDimensions));
        let res = super::same_storage_fast_stack(&[a.borrowed(), d.borrowed()]);
        assert_eq!(res, Err(IncompatibleStorages));
    }
}
