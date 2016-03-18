///! Sparse matrix product

use sparse::csmat::{CsMatOwned, CsMatView};
use sparse::vec::{CsVecView, CsVecOwned};
use num::traits::Num;
use sparse::compressed::SpMatView;
use ndarray::{ArrayView, ArrayViewMut, Axis};
use errors::SprsError;
use ::SpRes;
use ::Ix2;

/// Multiply a sparse CSC matrix with a dense vector and accumulate the result
/// into another dense vector
pub fn mul_acc_mat_vec_csc<N>(mat: CsMatView<N>,
                              in_vec: &[N],
                              res_vec: &mut[N]) -> SpRes<()>
where N: Num + Copy {
    let mat = mat.view();
    if mat.cols() != in_vec.len() || mat.rows() != res_vec.len() {
        return Err(SprsError::IncompatibleDimensions);
    }
    if !mat.is_csc() {
        return Err(SprsError::IncompatibleStorages);
    }

    for (col_ind, vec) in mat.outer_iterator() {
        let multiplier = &in_vec[col_ind];
        for (row_ind, &value) in vec.iter() {
            // TODO: unsafe access to value? needs bench
            res_vec[row_ind] =
                res_vec[row_ind] + *multiplier * value;
        }
    }
    Ok(())
}

/// Multiply a sparse CSR matrix with a dense vector and accumulate the result
/// into another dense vector
pub fn mul_acc_mat_vec_csr<N>(mat: CsMatView<N>,
                              in_vec: &[N],
                              res_vec: &mut[N]) -> SpRes<()>
where N: Num + Copy {
    if mat.cols() != in_vec.len() || mat.rows() != res_vec.len() {
        return Err(SprsError::IncompatibleDimensions);
    }
    if !mat.is_csr() {
        return Err(SprsError::IncompatibleStorages);
    }

    for (row_ind, vec) in mat.outer_iterator() {
        for (col_ind, &value) in vec.iter() {
            // TODO: unsafe access to value? needs bench
            res_vec[row_ind] =
                res_vec[row_ind] + in_vec[col_ind] * value;
        }
    }
    Ok(())
}


/// Perform a matrix multiplication for matrices sharing the same storage order.
///
/// For brevity, this method assumes a CSR storage order, transposition should
/// be used for the CSC-CSC case.
/// Accumulates the result line by line.
///
/// lhs: left hand size matrix
/// rhs: right hand size matrix
/// workspace: used to accumulate the line values. Should be of length
///            rhs.cols()
pub fn csr_mul_csr<N, Mat1, Mat2>(lhs: &Mat1,
                                  rhs: &Mat2,
                                  workspace: &mut[N]
                                 ) -> SpRes<CsMatOwned<N>>
where
N: Num + Copy,
Mat1: SpMatView<N>,
Mat2: SpMatView<N> {
    csr_mul_csr_impl(lhs.view(), rhs.view(), workspace)
}

/// Perform a matrix multiplication for matrices sharing the same storage order.
///
/// This method assumes a CSC storage order, and uses free transposition to
/// invoke the CSR method
///
/// lhs: left hand size matrix
/// rhs: right hand size matrix
/// workspace: used to accumulate the line values. Should be of length
///            lhs.lines()
pub fn csc_mul_csc<N, Mat1, Mat2>(lhs: &Mat1,
                                  rhs: &Mat2,
                                  workspace: &mut[N]
                                 ) -> SpRes<CsMatOwned<N>>
where
N: Num + Copy,
Mat1: SpMatView<N>,
Mat2: SpMatView<N> {
    csr_mul_csr_impl(rhs.transpose_view(),
                     lhs.transpose_view(),
                     workspace).map(|x| x.transpose_into())
}

/// Allocate the appropriate workspace for a CSR-CSR product
pub fn workspace_csr<N, Mat1, Mat2>(_: &Mat1, rhs: &Mat2) -> Vec<N>
where N: Copy + Num,
      Mat1: SpMatView<N>,
      Mat2: SpMatView<N> {
    let len = rhs.view().cols();
    vec![N::zero(); len]
}

/// Allocate the appropriate workspace for a CSC-CSC product
pub fn workspace_csc<N, Mat1, Mat2>(lhs: &Mat1, _: &Mat2) -> Vec<N>
where N: Copy + Num,
      Mat1: SpMatView<N>,
      Mat2: SpMatView<N> {
    let len = lhs.view().rows();
    vec![N::zero(); len]
}

/// Actual implementation of CSR-CSR multiplication
/// All other matrix products are implemented in terms of this one.
pub fn csr_mul_csr_impl<N>(lhs: CsMatView<N>,
                           rhs: CsMatView<N>,
                           workspace: &mut[N]
                          ) -> SpRes<CsMatOwned<N>>
where N: Num + Copy {
    let res_rows = lhs.rows();
    let res_cols = rhs.cols();
    if lhs.cols() != rhs.rows() {
        return Err(SprsError::IncompatibleDimensions);
    }
    if res_cols !=  workspace.len() {
        return Err(SprsError::BadWorkspaceDimensions);
    }
    if lhs.storage() != rhs.storage() {
        return Err(SprsError::IncompatibleStorages);
    }
    if !rhs.is_csr() {
        return Err(SprsError::BadStorageType);
    }

    let mut res = CsMatOwned::empty(lhs.storage(), res_cols);
    res.reserve_nnz_exact(lhs.nb_nonzero() + rhs.nb_nonzero());
    for (_, lvec) in lhs.outer_iterator() {
        // reset the accumulators
        for wval in workspace.iter_mut() {
            *wval = N::zero();
        }
        // accumulate the resulting row
        for (lcol, &lval) in lvec.iter() {
            // we can't be out of bounds thanks to the checks of dimension
            // compatibility and the structure check of CsMat. Therefore it
            // should be safe to call into an unsafe version of outer_view
            let rvec = rhs.outer_view(lcol).unwrap();
            for (rcol, &rval) in rvec.iter() {
                let wval = &mut workspace[rcol];
                let prod = lval * rval;
                *wval = *wval + prod;
            }
        }
        // compress the row into the resulting matrix
        res = res.append_outer(&workspace);
    }
    // TODO: shrink res storage? would need methods on CsMatOwned
    assert_eq!(res_rows, res.rows());
    Ok(res)
}

/// CSR-vector multiplication
pub fn csr_mul_csvec<N>(lhs: CsMatView<N>,
                        rhs: CsVecView<N>) -> SpRes<CsVecOwned<N>>
where N: Copy + Num {
    if lhs.cols() != rhs.dim() {
        return Err(SprsError::IncompatibleDimensions);
    }
    let mut res = CsVecOwned::empty(lhs.rows());
    for (row_ind, lvec) in lhs.outer_iterator() {
        let val = lvec.dot(&rhs);
        if val != N::zero() {
            res.append(row_ind, val);
        }
    }
    Ok(res)
}

/// CSR-dense rowmaj multiplication
///
/// Performs better if out is rowmaj.
pub fn csr_mulacc_dense_rowmaj<'a, N>(lhs: CsMatView<N>,
                                      rhs: ArrayView<N, Ix2>,
                                      mut out: ArrayViewMut<'a, N, Ix2>
                                     ) -> SpRes<()>
where N: 'a + Num + Copy
{
    if lhs.cols() != rhs.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if lhs.rows() != out.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if rhs.shape()[1] != out.shape()[1] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if !lhs.is_csr() {
        return Err(SprsError::BadStorageType);
    }
    if !rhs.is_standard_layout() {
        return Err(SprsError::BadStorageType);
    }
    // for now we implement a naive strategy, but later on it would
    // be nice to pick a data dependent block-size to optimize caching effects
    let lblock_size = 4;
    let rblock_size = 4;
    let axis0 = Axis(0);
    for (mut oblock, lblock) in out.axis_chunks_iter_mut(axis0, lblock_size)
                                   .zip(lhs.outer_block_iter(lblock_size)) {
        for (rcount, rblock) in rhs.axis_chunks_iter(axis0, rblock_size)
                                   .enumerate() {
            let col_start = rblock_size * rcount;
            let col_end = col_start + rblock_size;

            for ((_, line), mut oline) in lblock.outer_iterator()
                                          .zip(oblock.axis_iter_mut(axis0)) {
                'col_block: for (col_ind, &lval) in line.iter() {
                    if col_ind < col_start {
                        continue 'col_block;
                    }
                    if col_ind >= col_end {
                        break 'col_block;
                    }
                    let k_inblock = col_ind - col_start;
                    let rline = rblock.subview(axis0, k_inblock);
                    for (oval, &rval) in oline.iter_mut().zip(rline.iter()) {
                        let prev = *oval;
                        *oval = prev + lval * rval;
                    }
                }
            }
        }
    }
    Ok(())
}

/// CSC-dense rowmaj multiplication
///
/// Performs better if out is rowmaj
pub fn csc_mulacc_dense_rowmaj<'a, N>(lhs: CsMatView<N>,
                                      rhs: ArrayView<N, Ix2>,
                                      mut out: ArrayViewMut<'a, N, Ix2>
                                     ) -> SpRes<()>
where N: 'a + Num + Copy
{
    if lhs.cols() != rhs.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if lhs.rows() != out.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if rhs.shape()[1] != out.shape()[1] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if !lhs.is_csc() {
        return Err(SprsError::BadStorageType);
    }
    if !rhs.is_standard_layout() {
        return Err(SprsError::BadStorageType);
    }

    for ((_, lcol), rline) in lhs.outer_iterator().zip(rhs.outer_iter()) {
        for (orow, &lval) in lcol.iter() {
            let mut oline = out.row_mut(orow);
            for (oval, &rval) in oline.iter_mut().zip(rline.iter()) {
                let prev = *oval;
                *oval = prev + lval * rval;
            }
        }
    }
    Ok(())
}

/// CSC-dense colmaj multiplication
///
/// Performs better if out is colmaj
pub fn csc_mulacc_dense_colmaj<'a, N>(lhs: CsMatView<N>,
                                      rhs: ArrayView<N, Ix2>,
                                      mut out: ArrayViewMut<'a, N, Ix2>
                                     ) -> SpRes<()>
where N: 'a + Num + Copy
{
    if lhs.cols() != rhs.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if lhs.rows() != out.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if rhs.shape()[1] != out.shape()[1] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if !lhs.is_csc() {
        return Err(SprsError::BadStorageType);
    }
    if rhs.is_standard_layout() {
        return Err(SprsError::BadStorageType);
    }

    let axis1 = Axis(1);
    for (mut ocol, rcol) in out.axis_iter_mut(axis1).zip(rhs.axis_iter(axis1)) {
        for (rrow, lcol) in lhs.outer_iterator() {
            let rval = rcol[[rrow]];
            for (orow, &lval) in lcol.iter() {
                let prev = ocol[[orow]];
                ocol[[orow]] = prev + lval * rval;
            }
        }
    }
    Ok(())
}


/// CSR-dense colmaj multiplication
///
/// Performs better if out is colmaj
pub fn csr_mulacc_dense_colmaj<'a, N>(lhs: CsMatView<N>,
                                      rhs: ArrayView<N, Ix2>,
                                      mut out: ArrayViewMut<'a, N, Ix2>
                                     ) -> SpRes<()>
where N: 'a + Num + Copy
{
    if lhs.cols() != rhs.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if lhs.rows() != out.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if rhs.shape()[1] != out.shape()[1] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if !lhs.is_csr() {
        return Err(SprsError::BadStorageType);
    }
    if rhs.is_standard_layout() {
        return Err(SprsError::BadStorageType);
    }
    let axis1 = Axis(1);
    for (mut ocol, rcol) in out.axis_iter_mut(axis1).zip(rhs.axis_iter(axis1)) {
        for (orow, lrow) in lhs.outer_iterator() {
            let mut prev = ocol[[orow]];
            for (rrow, &lval) in lrow.iter() {
                let rval = rcol[[rrow]];
                prev = prev + lval * rval;
            }
            ocol[[orow]] = prev;
        }
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use sparse::csmat::{CsMat, CsMatOwned};
    use sparse::vec::{CsVec};
    use sparse::csmat::CompressedStorage::{CSC, CSR};
    use super::{mul_acc_mat_vec_csc, mul_acc_mat_vec_csr, csr_mul_csr};
    use test_data::{mat1, mat2, mat1_self_matprod, mat1_matprod_mat2,
                    mat1_csc, mat4, mat1_csc_matprod_mat4, mat_dense1,
                    mat5, mat_dense1_colmaj, mat_dense2};
    use ndarray::{OwnedArray, arr2};

    #[test]
    fn mul_csc_vec() {
        let indptr: &[usize] = &[0, 2, 4, 5, 6, 7];
        let indices: &[usize] = &[2, 3, 3, 4, 2, 1, 3];
        let data: &[f64] = &[
            0.35310881, 0.42380633, 0.28035896, 0.58082095,
            0.53350123, 0.88132896, 0.72527863];

        let mat = CsMat::new_borrowed(CSC, 5, 5, indptr, indices, data).unwrap();
        let vector = vec![0.1, 0.2, -0.1, 0.3, 0.9];
        let mut res_vec = vec![0., 0., 0., 0., 0.];
        mul_acc_mat_vec_csc(mat, &vector, &mut res_vec).unwrap();

        let expected_output =
            vec![ 0., 0.26439869, -0.01803924, 0.75120319, 0.11616419];

        let epsilon = 1e-7; // TODO: get better values and increase precision

        assert!(res_vec.iter().zip(expected_output.iter()).all(
            |(x,y)| (*x-*y).abs() < epsilon));
    }

    #[test]
    fn mul_csr_vec() {
        let indptr: &[usize] = &[0, 3, 3, 5, 6, 7];
        let indices: &[usize] = &[1, 2, 3, 2, 3, 4, 4];
        let data: &[f64] = &[
            0.75672424, 0.1649078, 0.30140296, 0.10358244,
            0.6283315, 0.39244208, 0.57202407];

        let mat = CsMat::new_borrowed(CSR, 5, 5, indptr, indices, data).unwrap();
        let vector = vec![0.1, 0.2, -0.1, 0.3, 0.9];
        let mut res_vec = vec![0., 0., 0., 0., 0.];
        mul_acc_mat_vec_csr(mat, &vector, &mut res_vec).unwrap();

        let expected_output =
            vec![0.22527496, 0., 0.17814121, 0.35319787, 0.51482166];

        let epsilon = 1e-7; // TODO: get better values and increase precision

        assert!(res_vec.iter().zip(expected_output.iter()).all(
            |(x,y)| (*x-*y).abs() < epsilon));
    }

    #[test]
    fn mul_csr_csr_identity() {
        let eye: CsMatOwned<i32> = CsMat::eye(CSR, 10);
        let mut workspace = [0; 10];
        let res = csr_mul_csr(&eye, &eye, &mut workspace).unwrap();
        assert_eq!(eye, res);

        let res = &eye * &eye;
        assert_eq!(eye, res);
    }

    #[test]
    fn mul_csr_csr() {
        let a = mat1();
        let res = &a * &a;
        let expected_output = mat1_self_matprod();
        assert_eq!(expected_output, res);

        let b = mat2();
        let res = &a * &b;
        let expected_output = mat1_matprod_mat2();
        assert_eq!(expected_output, res);
    }

    #[test]
    fn mul_csc_csc() {
        let a = mat1_csc();
        let b = mat4();
        let res = &a * &b;
        let expected_output = mat1_csc_matprod_mat4();
        assert_eq!(expected_output, res);

    }

    #[test]
    fn mul_csc_csr() {
        let a = mat1();
        let a_ = mat1_csc();
        let expected_output = mat1_self_matprod();

        let res = &a * &a_;
        assert_eq!(expected_output, res);

        let res = (&a_ * &a).to_other_storage();
        assert_eq!(expected_output, res);
    }

    #[test]
    fn mul_csr_csvec() {
        let a = mat1();
        let v = CsVec::new_owned(5, vec![0, 2, 4], vec![1.; 3]).unwrap();
        let res = &a * &v;
        let expected_output = CsVec::new_owned(5,
                                               vec![0, 1, 2],
                                               vec![3., 5., 5.]).unwrap();
        assert_eq!(expected_output, res);
    }

    #[test]
    fn mul_csvec_csr() {
        let a = mat1();
        let v = CsVec::new_owned(5, vec![0, 2, 4], vec![1.; 3]).unwrap();
        let res = &v * &a;
        let expected_output = CsVec::new_owned(5,
                                               vec![2, 3],
                                               vec![8., 11.]).unwrap();
        assert_eq!(expected_output, res);
    }

    #[test]
    fn mul_csc_csvec() {
        let a = mat1_csc();
        let v = CsVec::new_owned(5, vec![0, 2, 4], vec![1.; 3]).unwrap();
        let res = &a * &v;
        let expected_output = CsVec::new_owned(5,
                                               vec![0, 1, 2],
                                               vec![3., 5., 5.]).unwrap();
        assert_eq!(expected_output, res);
    }

    #[test]
    fn mul_csvec_csc() {
        let a = mat1_csc();
        let v = CsVec::new_owned(5, vec![0, 2, 4], vec![1.; 3]).unwrap();
        let res = &v * &a;
        let expected_output = CsVec::new_owned(5,
                                               vec![2, 3],
                                               vec![8., 11.]).unwrap();
        assert_eq!(expected_output, res);
    }

    #[test]
    fn mul_csr_dense_rowmaj() {
        let a = OwnedArray::eye(3);
        let e: CsMatOwned<f64> = CsMat::eye(CSR, 3);
        let mut res = OwnedArray::zeros((3, 3));
        super::csr_mulacc_dense_rowmaj(e.view(),
                                       a.view(),
                                       res.view_mut()
                                      ).unwrap();
        assert_eq!(res, a);

        let a = mat1();
        let b = mat_dense1();
        let mut res = OwnedArray::zeros((5, 5));
        super::csr_mulacc_dense_rowmaj(a.view(),
                                       b.view(),
                                       res.view_mut()
                                      ).unwrap();
        let expected_output = arr2(&[[24., 31., 24., 17., 10.],
                                     [11., 18., 11.,  9.,  2.],
                                     [20., 25., 20., 15., 10.],
                                     [40., 48., 40., 32., 24.],
                                     [21., 28., 21., 14.,  7.]]);
        assert_eq!(res, expected_output);

        let c = &a * &b;
        assert_eq!(c, expected_output);

        let a = mat5();
        let b = mat_dense2();
        let mut res = OwnedArray::zeros((5, 7));
        super::csr_mulacc_dense_rowmaj(a.view(),
                                       b.view(),
                                       res.view_mut()
                                      ).unwrap();
        let expected_output = arr2(
            &[[130.04,  150.1,  87.19, 90.89,  99.48,  80.43,   99.3],
              [217.72, 161.61,  79.47, 121.5, 124.23, 146.91, 157.79],
              [  55.6,  59.95,   86.7,   0.9,   37.4,  71.66,  51.94],
              [118.18, 123.16, 128.04, 92.02, 106.84,  175.1,  87.36],
              [  43.4,   54.1,  12.65, 44.35,   39.9,   23.4,   76.6]]);
        let eps = 1e-8;
        assert!(res.iter().zip(expected_output.iter())
                   .all(|(&x, &y)| (x - y).abs() <= eps));
    }

    #[test]
    fn mul_csc_dense_rowmaj() {
        let a = mat1_csc();
        let b = mat_dense1();
        let mut res = OwnedArray::zeros((5, 5));
        super::csc_mulacc_dense_rowmaj(a.view(),
                                       b.view(),
                                       res.view_mut()).unwrap();
        let expected_output = arr2(&[[24., 31., 24., 17., 10.],
                                     [11., 18., 11.,  9.,  2.],
                                     [20., 25., 20., 15., 10.],
                                     [40., 48., 40., 32., 24.],
                                     [21., 28., 21., 14., 7. ]]);
        assert_eq!(res, expected_output);

        let c = &a * &b;
        assert_eq!(c, expected_output);
    }

    #[test]
    fn mul_csc_dense_colmaj() {
        let a = mat1_csc();
        let b = mat_dense1_colmaj();
        let mut res = OwnedArray::zeros_f((5, 5));
        super::csc_mulacc_dense_colmaj(a.view(),
                                       b.view(),
                                       res.view_mut()).unwrap();
        let v = vec![24., 11., 20., 40., 21.,
                     31., 18., 25., 48., 28.,
                     24., 11., 20., 40., 21.,
                     17., 9., 15., 32., 14.,
                     10., 2., 10., 24., 7.];
        let expected_output = OwnedArray::from_vec_dim_stride((5, 5),
                                                              (1, 5),
                                                              v).unwrap();
        assert_eq!(res, expected_output);

        let c = &a * &b;
        assert_eq!(c, expected_output);
    }


    #[test]
    fn mul_csr_dense_colmaj() {
        let a = mat1();
        let b = mat_dense1_colmaj();
        let mut res = OwnedArray::zeros_f((5, 5));
        super::csr_mulacc_dense_colmaj(a.view(),
                                       b.view(),
                                       res.view_mut()
                                      ).unwrap();
        let v = vec![24., 11., 20., 40., 21.,
                    31., 18., 25., 48., 28.,
                    24., 11., 20., 40., 21.,
                    17., 9., 15., 32., 14.,
                    10., 2., 10., 24., 7.];
        let expected_output = OwnedArray::from_vec_dim_stride((5, 5),
                                                              (1, 5),
                                                              v).unwrap();
        assert_eq!(res, expected_output);

        let c = &a * &b;
        assert_eq!(c, expected_output);
    }
}
