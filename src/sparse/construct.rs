//! High level construction of sparse matrices by stacking, by block, ...

use std::default::Default;
use std::cmp;
use sparse::csmat::{CsMatVec, CsMatView};
use errors::SprsError;

/// Stack the given matrices into a new one, using the most efficient stacking
/// direction (ie vertical stack for CSR matrices, horizontal stack for CSC)
pub fn same_storage_fast_stack<'a, N, MatArray>(
    mats: &MatArray) -> Result<CsMatVec<N>, SprsError>
where N: 'a + Copy,
      MatArray: AsRef<[CsMatView<'a, N>]> {
    let mats = mats.as_ref();
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
pub fn vstack<'a, N, MatArray>(mats: &MatArray) -> Result<CsMatVec<N>, SprsError>
where N: 'a + Copy + Default,
      MatArray: AsRef<[CsMatView<'a, N>]> {
    let mats = mats.as_ref();
    if mats.iter().all(|x| x.is_csr()) {
        return same_storage_fast_stack(&mats);
    }

    let mats_csr: Vec<_> = mats.iter().map(|x| x.to_csr()).collect();
    let mats_csr_views: Vec<_> = mats_csr.iter().map(|x| x.borrowed()).collect();
    same_storage_fast_stack(&mats_csr_views)
}

/// Construct a sparse matrix by horizontally stacking other matrices
pub fn hstack<'a, N, MatArray>(mats: &MatArray) -> Result<CsMatVec<N>, SprsError>
where N: 'a + Copy + Default,
      MatArray: AsRef<[CsMatView<'a, N>]> {
    let mats = mats.as_ref();
    if mats.iter().all(|x| x.is_csc()) {
        return same_storage_fast_stack(&mats);
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
/// assert_eq!(c.rows(), 7);
/// ```
pub fn bmat<'a, N, OuterArray, InnerArray>(mats: &OuterArray)
-> Result<CsMatVec<N>, SprsError>
where N: 'a + Copy + Default,
      OuterArray: 'a + AsRef<[InnerArray]>,
      InnerArray: 'a + AsRef<[Option<CsMatView<'a, N>>]> {
    let mats = mats.as_ref();
    let super_rows = mats.len();
    if super_rows == 0 {
        return Err(SprsError::EmptyStackingList);
    }
    let super_cols = mats[0].as_ref().len();
    if super_cols == 0 {
        return Err(SprsError::EmptyStackingList);
    }

    // check input has matrix shape
    if ! mats.iter().all(|x| x.as_ref().len() == super_cols) {
        return Err(SprsError::IncompatibleDimensions);
    }

    if mats.iter().any(|x| x.as_ref().iter().all(|y| y.is_none())) {
        return Err(SprsError::EmptyBmatRow);
    }
    if (0..super_cols).any(|j| mats.iter().all(|x| x.as_ref()[j].is_none())) {
        return Err(SprsError::EmptyBmatCol);
    }

    // find out the shapes of the None elements
    let rows_per_row: Vec<_> = mats.iter().map(|row| {
        row.as_ref().iter().fold(0, |nrows, mopt| {
            mopt.as_ref().map_or(nrows, |m| cmp::max(nrows, m.rows()))
        })
    }).collect();
    let cols_per_col: Vec<_> = (0..super_cols).map(|j| {
        mats.iter().fold(0, |ncols, row| {
            row.as_ref()[j].as_ref()
                           .map_or(ncols, |m| cmp::max(ncols, m.cols()))
        })
    }).collect();
    let mut to_vstack = Vec::new();
    to_vstack.reserve(super_rows);
    for (i, row) in mats.iter().enumerate() {
        let with_zeros: Vec<_> = row.as_ref().iter().enumerate().map(|(j, m)| {
            m.as_ref().map_or(CsMatVec::zero(rows_per_row[i], cols_per_col[j]),
                              |x| x.to_owned())
        }).collect();
        let borrows: Vec<_> = with_zeros.iter().map(|x| x.borrowed()).collect();
        let stacked = try!(hstack(&borrows));
        to_vstack.push(stacked);
    }
    let borrows: Vec<_> = to_vstack.iter().map(|x| x.borrowed()).collect();
    vstack(&borrows)
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

    #[test]
    fn bmat_simple() {
        let a = CsMatVec::<f64>::eye(CSR, 5);
        let b = CsMatVec::<f64>::eye(CSR, 4);
        let c = super::bmat(&[[Some(a.borrowed()), None],
                              [None, Some(b.borrowed())]]).unwrap();
        let expected = CsMatVec::from_vecs(
            CSR, 9, 9,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            vec![1.; 9]).unwrap();
        assert_eq!(c, expected);
    }

    #[test]
    fn bmat_complex() {
        let a = mat1();
        let b = mat2();
        let c = super::bmat(&[[Some(a.borrowed()), Some(b.borrowed())],
                              [Some(b.borrowed()), None]]).unwrap();
        let expected = CsMatVec::from_vecs(
            CSR, 10, 10,
            vec![0,  6, 10, 11, 14, 17, 21, 23, 23, 25, 27],
            vec![2, 3, 5, 6, 7, 9, 3, 4, 5, 8, 2, 1, 7, 8, 3,
                 6, 7, 0, 1, 2, 4, 0, 3, 2, 3, 1, 2],
            vec![3., 4., 6., 7., 3., 3., 2., 5., 8., 9., 5., 8., 2., 4.,
                 7., 4., 4., 6., 7., 3., 3., 8., 9., 2., 4., 4., 4.]).unwrap();
        assert_eq!(c, expected);

        let d = mat3();
        let e = mat4();
        let f = super::bmat(&[[Some(d.borrowed()), Some(a.borrowed())],
                              [None, Some(e.borrowed())]]
                           ).unwrap();
        let expected = CsMatVec::from_vecs(
            CSR, 10, 9,
            vec![0, 4, 8, 10, 12, 14, 16, 18, 21, 23, 24],
            vec![2, 3, 6, 7, 2, 3, 7, 8, 2, 6, 1, 5, 3, 7, 4,
                 5, 4, 8, 4, 7, 8, 5, 7, 4],
            vec![3., 4., 3., 4., 2., 5., 2., 5., 5., 5., 8., 8.,
                 7., 7., 6., 8., 7., 4., 3., 2., 4., 9., 4., 3.]).unwrap();
        assert_eq!(f, expected);
    }
}
