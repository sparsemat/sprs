//! High level construction of sparse matrices by stacking, by block, ...

use std::ops::{Deref};
use std::default::Default;
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

/// Construct a sparse matrix by vertically stacking other matrices
pub fn vstack<N>(mats: &[CsMatView<N>]) -> Result<CsMatVec<N>, SprsError>
where N: Copy + Default {
    if mats.iter().all(|x| x.is_csr()) {
        return same_storage_fast_stack(mats);
    }

    let mats_csr: Vec<_> = mats.iter().map(|x| x.to_csr()).collect();
    let mats_csr_views: Vec<_> = mats_csr.iter().map(|x| x.borrowed()).collect();
    same_storage_fast_stack(&mats_csr_views)
}

/// Construct a sparse matrix by horizontally stacking other matrices
pub fn hstack<N>(mats: &[CsMatView<N>]) -> Result<CsMatVec<N>, SprsError>
where N: Copy + Default {
    if mats.iter().all(|x| x.is_csc()) {
        return same_storage_fast_stack(mats);
    }

    let mats_csc: Vec<_> = mats.iter().map(|x| x.to_csc()).collect();
    let mats_csc_views: Vec<_> = mats_csc.iter().map(|x| x.borrowed()).collect();
    same_storage_fast_stack(&mats_csc_views)
}

/// Specify a sparse matrix by constructing it from blocks of other matrices
///
/// # Examples
/// ```
/// use sprs::sparse::CompressedStorage::CSR;
/// use sprs::CsMatVec;
/// let a = CsMatVec::<f64>::eye(CSR, 3);
/// let b = CsMatVec::<f64>::eye(CSR, 4);
/// let c = sprs::bmat(&[[Some(a.borrowed()), None],
///                      [None, Some(b.borrowed())]]).unwrap();
/// assert_eq!(c.rows(), 5);
/// ```
pub fn bmat<'a, N, OuterArray, InnerArray>(mats: &OuterArray)
-> Result<CsMatVec<N>, SprsError>
where N: 'a + Copy + Default,
      OuterArray: 'a + AsRef<[InnerArray]>,
      InnerArray: 'a + AsRef<[Option<CsMatView<'a, N>>]> {
    let mats = mats.as_ref();
    let rows = mats.len();
    if rows == 0 {
        return Err(SprsError::EmptyStackingList);
    }
    let cols = mats[0].as_ref().len();
    if cols == 0 {
        return Err(SprsError::EmptyStackingList);
    }
    // check input has matrix shape
    if ! mats.iter().all(|x| x.as_ref().len() == cols) {
        return Err(SprsError::IncompatibleDimensions);
    }

    if mats.iter().any(|x| x.as_ref().iter().all(|y| y.is_none())) {
        return Err(SprsError::EmptyBmatRow);
    }
    if (0..cols).any(|i| mats.iter().all(|x| x.as_ref()[i].is_none())) {
        return Err(SprsError::EmptyBmatCol);
    }
    // start by checking if our input is well formed (no column or line of None)
    unimplemented!();
}

#[cfg(test)]
mod test {
    use sparse::csmat::CsMatVec;
    use sparse::CompressedStorage::{CSR};
    use test_data::{mat1, mat2, mat3, mat4};
    use errors::SprsError::*;

    fn mat1_vstack_mat2() -> CsMatVec<f64> {
        let indptr = vec![0, 2, 4, 5, 6, 7, 11, 13, 13, 15, 17];
        let indices = vec![2, 3, 3, 4, 2, 1, 3, 0, 1, 2, 4, 0, 3, 2, 3, 1, 2];
        let data = vec![3., 4., 2., 5., 5., 8., 7., 6., 7., 3., 3.,
                        8., 9., 2., 4., 4., 4.];
        CsMatVec::from_vecs(CSR, 10, 5, indptr, indices, data).unwrap()
    }

    #[test]
    fn same_storage_fast_stack_failures() {
        let res: Result<CsMatVec<f64>, _> =
            super::same_storage_fast_stack(&[]);
        assert_eq!(res, Err(EmptyStackingList));
        let a = mat1();
        let c = mat3();
        let d = mat4();
        let res: Result<CsMatVec<f64>, _> =
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

    #[test]
    fn vstack_trivial() {
        let a = mat1();
        let b = mat2();
        let res = super::vstack(&[a.borrowed(), b.borrowed()]);
        let expected = mat1_vstack_mat2();
        assert_eq!(res, Ok(expected));
    }

    #[test]
    fn hstack_trivial() {
        let a = mat1().transpose_into();
        let b = mat2().transpose_into();
        let res = super::hstack(&[a.borrowed(), b.borrowed()]);
        let expected = mat1_vstack_mat2().transpose_into();
        assert_eq!(res, Ok(expected));
    }

    #[test]
    fn vstack_with_conversion() {
        let a = mat1().to_csc();
        let b = mat2();
        let res = super::vstack(&[a.borrowed(), b.borrowed()]);
        let expected = mat1_vstack_mat2();
        assert_eq!(res, Ok(expected));
    }

    #[test]
    fn bmat_failures() {
        let res: Result<CsMatVec<f64>, _> =
            super::bmat(&[[]]);
        assert_eq!(res, Err(EmptyStackingList));
        let a = mat1();
        let c = mat3();
        let res: Result<CsMatVec<f64>,_> = super::bmat(
            &vec![vec![None, None], vec![None]]);
        assert_eq!(res, Err(IncompatibleDimensions));
        let res: Result<CsMatVec<f64>, _> =
            super::bmat(&[[None, None],
                          [Some(a.borrowed()), Some(c.borrowed())]]);
        assert_eq!(res, Err(EmptyBmatRow));
        let res: Result<CsMatVec<f64>, _> =
            super::bmat(&[[Some(c.borrowed()), None],
                          [Some(a.borrowed()), None]]);
        assert_eq!(res, Err(EmptyBmatCol));
    }
}
