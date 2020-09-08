use crate::indexing::SpIndex;
use crate::sparse::compressed::SpMatView;
///! Sparse matrix product
use crate::sparse::prelude::*;
use crate::sparse::vec::DenseVector;
use crate::Ix2;
use ndarray::{ArrayView, ArrayViewMut, Axis};
use num_traits::Num;
use std::iter::Sum;

/// Compute the dot product of two sparse vectors, using binary search to find matching indices.
///
/// Runs in O(MlogN) time, where M and N are the number of non-zero entries in each vector.
pub fn csvec_dot_by_binary_search<N, I>(
    vec1: CsVecViewI<N, I>,
    vec2: CsVecViewI<N, I>,
) -> N
where
    I: SpIndex,
    N: Num + Copy,
{
    let (mut idx1, mut val1, mut idx2, mut val2) = if vec1.nnz() < vec2.nnz() {
        (vec1.indices(), vec1.data(), vec2.indices(), vec2.data())
    } else {
        (vec2.indices(), vec2.data(), vec1.indices(), vec1.data())
    };

    let mut sum = N::zero();
    while !idx1.is_empty() && !idx2.is_empty() {
        debug_assert_eq!(idx1.len(), val1.len());
        debug_assert_eq!(idx2.len(), val2.len());

        let (found, i) = match idx2.binary_search(&idx1[0]) {
            Ok(i) => (true, i),
            Err(i) => (false, i),
        };
        if found {
            sum = sum + val1[0] * val2[i];
        }
        idx1 = &idx1[1..];
        val1 = &val1[1..];
        idx2 = &idx2[i..];
        val2 = &val2[i..];
    }
    sum
}

/// Multiply a sparse CSC matrix with a dense vector and accumulate the result
/// into another dense vector
pub fn mul_acc_mat_vec_csc<N, I, Iptr, V>(
    mat: CsMatViewI<N, I, Iptr>,
    in_vec: V,
    res_vec: &mut [N],
) where
    N: Num + Copy,
    I: SpIndex,
    Iptr: SpIndex,
    V: DenseVector<N>,
{
    let mat = mat.view();
    if mat.cols() != in_vec.dim() || mat.rows() != res_vec.len() {
        panic!("Dimension mismatch");
    }
    if !mat.is_csc() {
        panic!("Storage mismatch");
    }

    for (col_ind, vec) in mat.outer_iterator().enumerate() {
        let multiplier = in_vec.index(col_ind);
        for (row_ind, &value) in vec.iter() {
            // TODO: unsafe access to value? needs bench
            res_vec[row_ind] = res_vec[row_ind] + *multiplier * value;
        }
    }
}

/// Multiply a sparse CSR matrix with a dense vector and accumulate the result
/// into another dense vector
pub fn mul_acc_mat_vec_csr<N, I, Iptr, V>(
    mat: CsMatViewI<N, I, Iptr>,
    in_vec: V,
    res_vec: &mut [N],
) where
    N: Num + Copy,
    I: SpIndex,
    Iptr: SpIndex,
    V: DenseVector<N>,
{
    if mat.cols() != in_vec.dim() || mat.rows() != res_vec.len() {
        panic!("Dimension mismatch");
    }
    if !mat.is_csr() {
        panic!("Storage mismatch");
    }

    for (row_ind, vec) in mat.outer_iterator().enumerate() {
        // this unwrap is ok because we did the check before to ensure
        // mat.row() == res_vec.len() and now the row_ind is within the
        // range of [0, mat.row). So it should be safe.
        let tv = res_vec.get_mut(row_ind).unwrap();
        for (col_ind, &value) in vec.iter() {
            // TODO: unsafe access to value? needs bench
            *tv = *tv + *in_vec.index(col_ind) * value;
        }
    }
}

/// Allocate the appropriate workspace for a CSR-CSR product
pub fn workspace_csr<N, I, Iptr, Mat1, Mat2>(_: &Mat1, rhs: &Mat2) -> Vec<N>
where
    N: Copy + Num,
    I: SpIndex,
    Iptr: SpIndex,
    Mat1: SpMatView<N, I, Iptr>,
    Mat2: SpMatView<N, I, Iptr>,
{
    let len = rhs.view().cols();
    vec![N::zero(); len]
}

/// Allocate the appropriate workspace for a CSC-CSC product
pub fn workspace_csc<N, I, Iptr, Mat1, Mat2>(lhs: &Mat1, _: &Mat2) -> Vec<N>
where
    N: Copy + Num,
    I: SpIndex,
    Iptr: SpIndex,
    Mat1: SpMatView<N, I, Iptr>,
    Mat2: SpMatView<N, I, Iptr>,
{
    let len = lhs.view().rows();
    vec![N::zero(); len]
}

/// CSR-vector multiplication
pub fn csr_mul_csvec<N, I, Iptr>(
    lhs: CsMatViewI<N, I, Iptr>,
    rhs: CsVecViewI<N, I>,
) -> CsVecI<N, I>
where
    N: Copy + Num + Sum,
    I: SpIndex,
    Iptr: SpIndex,
{
    if rhs.dim == 0 {
        return rhs.to_owned();
    }
    if lhs.cols() != rhs.dim() {
        panic!("Dimension mismatch");
    }
    let mut res = CsVecI::empty(lhs.rows());
    for (row_ind, lvec) in lhs.outer_iterator().enumerate() {
        let val = lvec.dot(&rhs);
        if val != N::zero() {
            res.append(row_ind, val);
        }
    }
    res
}

/// CSR-dense rowmaj multiplication
///
/// Performs better if rhs has a decent number of colums.
pub fn csr_mulacc_dense_rowmaj<'a, N1, N2, NOut, I, Iptr>(
    lhs: CsMatViewI<N1, I, Iptr>,
    rhs: ArrayView<N2, Ix2>,
    mut out: ArrayViewMut<'a, NOut, Ix2>,
) where
    N1: 'a + Num + Copy,
    N2: 'a + Num + Copy,
    NOut: 'a + Num + Copy,
    N1: std::ops::Mul<N2, Output = NOut>,
    I: 'a + SpIndex,
    Iptr: 'a + SpIndex,
{
    if lhs.cols() != rhs.shape()[0] {
        panic!("Dimension mismatch");
    }
    if lhs.rows() != out.shape()[0] {
        panic!("Dimension mismatch");
    }
    if rhs.shape()[1] != out.shape()[1] {
        panic!("Dimension mismatch");
    }
    if !lhs.is_csr() {
        panic!("Storage mismatch");
    }

    let axis0 = Axis(0);
    for (line, mut oline) in lhs.outer_iterator().zip(out.axis_iter_mut(axis0))
    {
        for (col_ind, &lval) in line.iter() {
            let rline = rhs.row(col_ind);
            // TODO: call an axpy primitive to benefit from vectorisation?
            for (oval, &rval) in oline.iter_mut().zip(rline.iter()) {
                let prev = *oval;
                *oval = prev + lval * rval;
            }
        }
    }
}

/// CSC-dense rowmaj multiplication
///
/// Performs better if rhs has a decent number of colums.
pub fn csc_mulacc_dense_rowmaj<'a, N1, N2, NOut, I, Iptr>(
    lhs: CsMatViewI<N1, I, Iptr>,
    rhs: ArrayView<N2, Ix2>,
    mut out: ArrayViewMut<'a, NOut, Ix2>,
) where
    N1: 'a + Num + Copy,
    N2: 'a + Num + Copy,
    NOut: 'a + Num + Copy,
    N1: std::ops::Mul<N2, Output = NOut>,
    I: 'a + SpIndex,
    Iptr: 'a + SpIndex,
{
    if lhs.cols() != rhs.shape()[0] {
        panic!("Dimension mismatch");
    }
    if lhs.rows() != out.shape()[0] {
        panic!("Dimension mismatch");
    }
    if rhs.shape()[1] != out.shape()[1] {
        panic!("Dimension mismatch");
    }
    if !lhs.is_csc() {
        panic!("Storage mismatch");
    }

    for (lcol, rline) in lhs.outer_iterator().zip(rhs.outer_iter()) {
        for (orow, &lval) in lcol.iter() {
            let mut oline = out.row_mut(orow);
            for (oval, &rval) in oline.iter_mut().zip(rline.iter()) {
                let prev = *oval;
                *oval = prev + lval * rval;
            }
        }
    }
}

/// CSC-dense colmaj multiplication
///
/// Performs better if rhs has few columns.
pub fn csc_mulacc_dense_colmaj<'a, N1, N2, NOut, I, Iptr>(
    lhs: CsMatViewI<N1, I, Iptr>,
    rhs: ArrayView<N2, Ix2>,
    mut out: ArrayViewMut<'a, NOut, Ix2>,
) where
    N1: 'a + Num + Copy,
    N2: 'a + Num + Copy,
    NOut: 'a + Num + Copy,
    N1: std::ops::Mul<N2, Output = NOut>,
    I: 'a + SpIndex,
    Iptr: 'a + SpIndex,
{
    if lhs.cols() != rhs.shape()[0] {
        panic!("Dimension mismatch");
    }
    if lhs.rows() != out.shape()[0] {
        panic!("Dimension mismatch");
    }
    if rhs.shape()[1] != out.shape()[1] {
        panic!("Dimension mismatch");
    }
    if !lhs.is_csc() {
        panic!("Storage mismatch");
    }

    let axis1 = Axis(1);
    for (mut ocol, rcol) in out.axis_iter_mut(axis1).zip(rhs.axis_iter(axis1)) {
        for (rrow, lcol) in lhs.outer_iterator().enumerate() {
            let rval = rcol[[rrow]];
            for (orow, &lval) in lcol.iter() {
                let prev = ocol[[orow]];
                ocol[[orow]] = prev + lval * rval;
            }
        }
    }
}

/// CSR-dense colmaj multiplication
///
/// Performs better if rhs has few columns.
pub fn csr_mulacc_dense_colmaj<'a, N1, N2, NOut, I, Iptr>(
    lhs: CsMatViewI<N1, I, Iptr>,
    rhs: ArrayView<N2, Ix2>,
    mut out: ArrayViewMut<'a, NOut, Ix2>,
) where
    N1: 'a + Num + Copy,
    N2: 'a + Num + Copy,
    NOut: 'a + Num + Copy,
    N1: std::ops::Mul<N2, Output = NOut>,
    I: 'a + SpIndex,
    Iptr: 'a + SpIndex,
{
    if lhs.cols() != rhs.shape()[0] {
        panic!("Dimension mismatch");
    }
    if lhs.rows() != out.shape()[0] {
        panic!("Dimension mismatch");
    }
    if rhs.shape()[1] != out.shape()[1] {
        panic!("Dimension mismatch");
    }
    if !lhs.is_csr() {
        panic!("Storage mismatch");
    }
    let axis1 = Axis(1);
    for (mut ocol, rcol) in out.axis_iter_mut(axis1).zip(rhs.axis_iter(axis1)) {
        for (orow, lrow) in lhs.outer_iterator().enumerate() {
            let mut prev = ocol[[orow]];
            for (rrow, &lval) in lrow.iter() {
                let rval = rcol[[rrow]];
                prev = prev + lval * rval;
            }
            ocol[[orow]] = prev;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::sparse::csmat::CompressedStorage::{CSC, CSR};
    use crate::sparse::{CsMat, CsMatView, CsVec};
    use crate::test_data::{
        mat1, mat1_csc, mat1_csc_matprod_mat4, mat1_matprod_mat2,
        mat1_self_matprod, mat2, mat4, mat5, mat_dense1, mat_dense1_colmaj,
        mat_dense2,
    };
    use ndarray::linalg::Dot;
    use ndarray::{arr2, s, Array, Array2, Dimension, ShapeBuilder};

    #[test]
    fn test_csvec_dot_by_binary_search() {
        let vec1 = CsVecI::new(8, vec![0, 2, 4, 6], vec![1.; 4]);
        let vec2 = CsVecI::new(8, vec![1, 3, 5, 7], vec![2.; 4]);
        let vec3 = CsVecI::new(8, vec![1, 2, 5, 6], vec![3.; 4]);

        assert_eq!(0., csvec_dot_by_binary_search(vec1.view(), vec2.view()));
        assert_eq!(4., csvec_dot_by_binary_search(vec1.view(), vec1.view()));
        assert_eq!(16., csvec_dot_by_binary_search(vec2.view(), vec2.view()));
        assert_eq!(6., csvec_dot_by_binary_search(vec1.view(), vec3.view()));
        assert_eq!(12., csvec_dot_by_binary_search(vec2.view(), vec3.view()));
    }

    #[test]
    fn mul_csc_vec() {
        let indptr: &[usize] = &[0, 2, 4, 5, 6, 7];
        let indices: &[usize] = &[2, 3, 3, 4, 2, 1, 3];
        let data: &[f64] = &[
            0.35310881, 0.42380633, 0.28035896, 0.58082095, 0.53350123,
            0.88132896, 0.72527863,
        ];

        let mat =
            CsMatView::new_view(CSC, (5, 5), indptr, indices, data).unwrap();
        let vector = vec![0.1, 0.2, -0.1, 0.3, 0.9];
        let mut res_vec = vec![0., 0., 0., 0., 0.];
        mul_acc_mat_vec_csc(mat, &vector, &mut res_vec);

        let expected_output =
            vec![0., 0.26439869, -0.01803924, 0.75120319, 0.11616419];

        let epsilon = 1e-7; // TODO: get better values and increase precision

        assert!(res_vec
            .iter()
            .zip(expected_output.iter())
            .all(|(x, y)| (*x - *y).abs() < epsilon));
    }

    #[test]
    fn mul_csr_vec() {
        let indptr: &[usize] = &[0, 3, 3, 5, 6, 7];
        let indices: &[usize] = &[1, 2, 3, 2, 3, 4, 4];
        let data: &[f64] = &[
            0.75672424, 0.1649078, 0.30140296, 0.10358244, 0.6283315,
            0.39244208, 0.57202407,
        ];

        let mat =
            CsMatView::new_view(CSR, (5, 5), indptr, indices, data).unwrap();
        let slice: &[f64] = &[0.1, 0.2, -0.1, 0.3, 0.9];
        let mut res_vec = vec![0., 0., 0., 0., 0.];
        mul_acc_mat_vec_csr(mat, slice, &mut res_vec);

        let expected_output =
            vec![0.22527496, 0., 0.17814121, 0.35319787, 0.51482166];

        let epsilon = 1e-7; // TODO: get better values and increase precision

        assert!(res_vec
            .iter()
            .zip(expected_output.iter())
            .all(|(x, y)| (*x - *y).abs() < epsilon));
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
        let v = CsVec::new(5, vec![0, 2, 4], vec![1.; 3]);
        let res = &a * &v;
        let expected_output = CsVec::new(5, vec![0, 1, 2], vec![3., 5., 5.]);
        assert_eq!(expected_output, res);
    }

    #[test]
    fn mul_csr_zero_csvec() {
        let zero = CsVec::new(0, vec![], vec![]);
        assert_eq!(&mat1() * &zero, zero);
    }

    #[test]
    fn mul_csvec_csr() {
        let a = mat1();
        let v = CsVec::new(5, vec![0, 2, 4], vec![1.; 3]);
        let res = &v * &a;
        let expected_output = CsVec::new(5, vec![2, 3], vec![8., 11.]);
        assert_eq!(expected_output, res);
    }

    #[test]
    fn mul_csc_csvec() {
        let a = mat1_csc();
        let v = CsVec::new(5, vec![0, 2, 4], vec![1.; 3]);
        let res = &a * &v;
        let expected_output = CsVec::new(5, vec![0, 1, 2], vec![3., 5., 5.]);
        assert_eq!(expected_output, res);
    }

    #[test]
    fn mul_csvec_csc() {
        let a = mat1_csc();
        let v = CsVec::new(5, vec![0, 2, 4], vec![1.; 3]);
        let res = &v * &a;
        let expected_output = CsVec::new(5, vec![2, 3], vec![8., 11.]);
        assert_eq!(expected_output, res);
    }

    #[test]
    fn mul_csr_dense_rowmaj() {
        let a: Array2<f64> = Array::eye(3);
        let e: CsMat<f64> = CsMat::eye(3);
        let mut res = Array::zeros((3, 3));
        super::csr_mulacc_dense_rowmaj(e.view(), a.view(), res.view_mut());
        assert_eq!(res, a);

        let a = mat1();
        let b = mat_dense1();
        let mut res = Array::zeros((5, 5));
        super::csr_mulacc_dense_rowmaj(a.view(), b.view(), res.view_mut());
        let expected_output = arr2(&[
            [24., 31., 24., 17., 10.],
            [11., 18., 11., 9., 2.],
            [20., 25., 20., 15., 10.],
            [40., 48., 40., 32., 24.],
            [21., 28., 21., 14., 7.],
        ]);
        assert_eq!(res, expected_output);

        let c = &a * &b;
        assert_eq!(c, expected_output);

        let a = mat5();
        let b = mat_dense2();
        let mut res = Array::zeros((5, 7));
        super::csr_mulacc_dense_rowmaj(a.view(), b.view(), res.view_mut());
        let expected_output = arr2(&[
            [130.04, 150.1, 87.19, 90.89, 99.48, 80.43, 99.3],
            [217.72, 161.61, 79.47, 121.5, 124.23, 146.91, 157.79],
            [55.6, 59.95, 86.7, 0.9, 37.4, 71.66, 51.94],
            [118.18, 123.16, 128.04, 92.02, 106.84, 175.1, 87.36],
            [43.4, 54.1, 12.65, 44.35, 39.9, 23.4, 76.6],
        ]);
        let eps = 1e-8;
        assert!(res
            .iter()
            .zip(expected_output.iter())
            .all(|(&x, &y)| (x - y).abs() <= eps));
    }

    #[test]
    fn mul_csc_dense_rowmaj() {
        let a = mat1_csc();
        let b = mat_dense1();
        let mut res = Array::zeros((5, 5));
        super::csc_mulacc_dense_rowmaj(a.view(), b.view(), res.view_mut());
        let expected_output = arr2(&[
            [24., 31., 24., 17., 10.],
            [11., 18., 11., 9., 2.],
            [20., 25., 20., 15., 10.],
            [40., 48., 40., 32., 24.],
            [21., 28., 21., 14., 7.],
        ]);
        assert_eq!(res, expected_output);

        let c = &a * &b;
        assert_eq!(c, expected_output);
    }

    #[test]
    fn mul_csc_dense_colmaj() {
        let a = mat1_csc();
        let b = mat_dense1_colmaj();
        let mut res = Array::zeros((5, 5).f());
        super::csc_mulacc_dense_colmaj(a.view(), b.view(), res.view_mut());
        let v = vec![
            24., 11., 20., 40., 21., 31., 18., 25., 48., 28., 24., 11., 20.,
            40., 21., 17., 9., 15., 32., 14., 10., 2., 10., 24., 7.,
        ];
        let expected_output = Array::from_shape_vec((5, 5).f(), v).unwrap();
        assert_eq!(res, expected_output);

        let c = &a * &b;
        assert_eq!(c, expected_output);
    }

    #[test]
    fn mul_csr_dense_colmaj() {
        let a = mat1();
        let b = mat_dense1_colmaj();
        let mut res = Array::zeros((5, 5).f());
        super::csr_mulacc_dense_colmaj(a.view(), b.view(), res.view_mut());
        let v = vec![
            24., 11., 20., 40., 21., 31., 18., 25., 48., 28., 24., 11., 20.,
            40., 21., 17., 9., 15., 32., 14., 10., 2., 10., 24., 7.,
        ];
        let expected_output = Array::from_shape_vec((5, 5).f(), v).unwrap();
        assert_eq!(res, expected_output);

        let c = &a * &b;
        assert_eq!(c, expected_output);
    }

    // stolen from ndarray - not currently exported.
    fn assert_close<D>(a: ArrayView<f64, D>, b: ArrayView<f64, D>)
    where
        D: Dimension,
    {
        let diff = (&a - &b).mapv_into(f64::abs);

        let rtol = 1e-7;
        let atol = 1e-12;
        let crtol = b.mapv(|x| x.abs() * rtol);
        let tol = crtol + atol;
        let tol_m_diff = &diff - &tol;
        let maxdiff = tol_m_diff.fold(0. / 0., |x, y| f64::max(x, *y));
        println!("diff offset from tolerance level= {:.2e}", maxdiff);
        if maxdiff > 0. {
            println!("{:.4?}", a);
            println!("{:.4?}", b);
            panic!("results differ");
        }
    }

    #[test]
    fn test_sparse_dot_dense() {
        let sparse = [
            mat1(),
            mat1_csc(),
            mat2(),
            mat2().transpose_into(),
            mat4(),
            mat5(),
        ];
        let dense = [
            mat_dense1(),
            mat_dense1_colmaj(),
            mat_dense1().reversed_axes(),
            mat_dense2(),
            mat_dense2().reversed_axes(),
        ];

        // test sparse.dot(dense)
        for s in sparse.iter() {
            for d in dense.iter() {
                if d.shape()[0] < s.cols() {
                    continue;
                }

                let d = d.slice(s![0..s.cols(), ..]);

                let truth = s.to_dense().dot(&d);
                let test = s.dot(&d);
                assert_close(test.view(), truth.view());
            }
        }
    }

    #[test]
    fn test_dense_dot_sparse() {
        let sparse = [
            mat1(),
            mat1_csc(),
            mat2(),
            mat2().transpose_into(),
            mat4(),
            mat5(),
        ];
        let dense = [
            mat_dense1(),
            mat_dense1_colmaj(),
            mat_dense1().reversed_axes(),
            mat_dense2(),
            mat_dense2().reversed_axes(),
        ];

        // test sparse.ldot(dense)
        for s in sparse.iter() {
            for d in dense.iter() {
                if d.shape()[1] < s.rows() {
                    continue;
                }

                let d = d.slice(s![.., 0..s.rows()]);

                let truth = d.dot(&s.to_dense());
                let test = d.dot(s);
                assert_close(test.view(), truth.view());
            }
        }
    }
}
