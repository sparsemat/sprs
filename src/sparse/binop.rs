///! Sparse matrix addition, subtraction
use crate::indexing::SpIndex;
use crate::sparse::compressed::SpMatView;
use crate::sparse::csmat::CompressedStorage;
use crate::sparse::prelude::*;
use crate::sparse::vec::NnzEither::{Both, Left, Right};
use crate::sparse::vec::SparseIterTools;
use ndarray::{
    self, Array, ArrayBase, ArrayView, ArrayViewMut, Axis, ShapeBuilder,
};
use num_traits::Num;

use crate::Ix2;
use crate::SpRes;

/// Sparse matrix addition, with matrices sharing the same storage type
pub fn add_mat_same_storage<N, I, Iptr, Mat1, Mat2>(
    lhs: &Mat1,
    rhs: &Mat2,
) -> CsMatI<N, I, Iptr>
where
    N: Num + Copy,
    I: SpIndex,
    Iptr: SpIndex,
    Mat1: SpMatView<N, I, Iptr>,
    Mat2: SpMatView<N, I, Iptr>,
{
    csmat_binop(lhs.view(), rhs.view(), |&x, &y| x + y)
}

/// Sparse matrix subtraction, with same storage type
pub fn sub_mat_same_storage<N, I, Iptr, Mat1, Mat2>(
    lhs: &Mat1,
    rhs: &Mat2,
) -> CsMatI<N, I, Iptr>
where
    N: Num + Copy,
    I: SpIndex,
    Iptr: SpIndex,
    Mat1: SpMatView<N, I, Iptr>,
    Mat2: SpMatView<N, I, Iptr>,
{
    csmat_binop(lhs.view(), rhs.view(), |&x, &y| x - y)
}

/// Sparse matrix scalar multiplication, with same storage type
pub fn mul_mat_same_storage<N, I, Iptr, Mat1, Mat2>(
    lhs: &Mat1,
    rhs: &Mat2,
) -> CsMatI<N, I, Iptr>
where
    N: Num + Copy,
    I: SpIndex,
    Iptr: SpIndex,
    Mat1: SpMatView<N, I, Iptr>,
    Mat2: SpMatView<N, I, Iptr>,
{
    csmat_binop(lhs.view(), rhs.view(), |&x, &y| x * y)
}

/// Sparse matrix multiplication by a scalar
pub fn scalar_mul_mat<N, I, Iptr, Mat>(mat: &Mat, val: N) -> CsMatI<N, I, Iptr>
where
    N: Num + Copy,
    I: SpIndex,
    Iptr: SpIndex,
    Mat: SpMatView<N, I, Iptr>,
{
    let mat = mat.view();
    mat.map(|&x| x * val)
}

/// Applies a binary operation to matching non-zero elements
/// of two sparse matrices. When e.g. only the `lhs` has a non-zero at a
/// given location, `0` is inferred for the non-zero value of the other matrix.
/// Both matrices should have the same storage.
///
/// Thus the behaviour is correct iff `binop(N::zero(), N::zero()) == N::zero()`
///
/// # Errors
///
/// - on incompatible dimensions
/// - on incomatible storage
pub fn csmat_binop<N, I, Iptr, F>(
    lhs: CsMatViewI<N, I, Iptr>,
    rhs: CsMatViewI<N, I, Iptr>,
    binop: F,
) -> CsMatI<N, I, Iptr>
where
    N: Num + Clone,
    I: SpIndex,
    Iptr: SpIndex,
    F: Fn(&N, &N) -> N,
{
    let nrows = lhs.rows();
    let ncols = lhs.cols();
    let storage = lhs.storage();
    if nrows != rhs.rows() || ncols != rhs.cols() {
        panic!("Dimension mismatch");
    }
    if storage != rhs.storage() {
        panic!("Storage mismatch");
    }

    let max_nnz = lhs.nnz() + rhs.nnz();
    let mut out_indptr = vec![Iptr::zero(); lhs.outer_dims() + 1];
    let mut out_indices = vec![I::zero(); max_nnz];

    // Sadly the vec! macro requires Clone, but we don't want to force
    // Clone on our consumers, so we have to use this workaround.
    // This should compile to decent code however.
    let mut out_data = vec![N::zero(); max_nnz];

    let nnz = csmat_binop_same_storage_raw(
        lhs,
        rhs,
        binop,
        &mut out_indptr[..],
        &mut out_indices[..],
        &mut out_data[..],
    );
    out_indices.truncate(nnz);
    out_data.truncate(nnz);
    CsMatI {
        storage,
        nrows,
        ncols,
        indptr: out_indptr,
        indices: out_indices,
        data: out_data,
    }
}

/// Raw implementation of scalar binary operation for compressed sparse matrices
/// sharing the same storage. The output arrays are assumed to be preallocated
///
/// Returns the nnz count
pub fn csmat_binop_same_storage_raw<N, I, Iptr, F>(
    lhs: CsMatViewI<N, I, Iptr>,
    rhs: CsMatViewI<N, I, Iptr>,
    binop: F,
    out_indptr: &mut [Iptr],
    out_indices: &mut [I],
    out_data: &mut [N],
) -> usize
where
    N: Num,
    I: SpIndex,
    Iptr: SpIndex,
    F: Fn(&N, &N) -> N,
{
    assert_eq!(lhs.cols(), rhs.cols());
    assert_eq!(lhs.rows(), rhs.rows());
    assert_eq!(lhs.storage(), rhs.storage());
    assert_eq!(out_indptr.len(), rhs.outer_dims() + 1);
    let max_nnz = lhs.nnz() + rhs.nnz();
    assert!(out_data.len() >= max_nnz);
    assert!(out_indices.len() >= max_nnz);
    let mut nnz = 0;
    out_indptr[0] = Iptr::zero();
    let iter = lhs.outer_iterator().zip(rhs.outer_iterator()).enumerate();
    for (dim, (lv, rv)) in iter {
        for elem in lv.iter().nnz_or_zip(rv.iter()) {
            let (ind, binop_val) = match elem {
                Left((ind, val)) => (ind, binop(val, &N::zero())),
                Right((ind, val)) => (ind, binop(&N::zero(), val)),
                Both((ind, lval, rval)) => (ind, binop(lval, rval)),
            };
            if binop_val != N::zero() {
                out_indices[nnz] = I::from_usize_unchecked(ind);
                out_data[nnz] = binop_val;
                nnz += 1;
            }
        }
        out_indptr[dim + 1] = Iptr::from_usize(nnz);
    }
    nnz
}

/// Compute alpha * lhs + beta * rhs with lhs a sparse matrix and rhs dense
/// and alpha and beta scalars
pub fn add_dense_mat_same_ordering<N, I, Iptr, Mat, D>(
    lhs: &Mat,
    rhs: &ArrayBase<D, Ix2>,
    alpha: N,
    beta: N,
) -> Array<N, Ix2>
where
    N: Num + Copy,
    I: SpIndex,
    Iptr: SpIndex,
    Mat: SpMatView<N, I, Iptr>,
    D: ndarray::Data<Elem = N>,
{
    let shape = (rhs.shape()[0], rhs.shape()[1]);
    let mut res = if rhs.is_standard_layout() {
        Array::zeros(shape)
    } else {
        Array::zeros(shape.f())
    };
    csmat_binop_dense_raw(
        lhs.view(),
        rhs.view(),
        |&x, &y| alpha * x + beta * y,
        res.view_mut(),
    );
    res
}

/// Compute coeff wise `alpha * lhs * rhs` with `lhs` a sparse matrix,
/// `rhs` a dense matrix, and `alpha` a scalar
pub fn mul_dense_mat_same_ordering<N, I, Iptr, Mat, D>(
    lhs: &Mat,
    rhs: &ArrayBase<D, Ix2>,
    alpha: N,
) -> Array<N, Ix2>
where
    N: Num + Copy,
    I: SpIndex,
    Iptr: SpIndex,
    Mat: SpMatView<N, I, Iptr>,
    D: ndarray::Data<Elem = N>,
{
    let shape = (rhs.shape()[0], rhs.shape()[1]);
    let mut res = if rhs.is_standard_layout() {
        Array::zeros(shape)
    } else {
        Array::zeros(shape.f())
    };
    csmat_binop_dense_raw(
        lhs.view(),
        rhs.view(),
        |&x, &y| alpha * x * y,
        res.view_mut(),
    );
    res
}

/// Raw implementation of sparse/dense binary operations with the same
/// ordering
pub fn csmat_binop_dense_raw<'a, N, I, Iptr, F>(
    lhs: CsMatViewI<'a, N, I, Iptr>,
    rhs: ArrayView<'a, N, Ix2>,
    binop: F,
    mut out: ArrayViewMut<'a, N, Ix2>,
) where
    N: 'a + Num,
    I: 'a + SpIndex,
    Iptr: 'a + SpIndex,
    F: Fn(&N, &N) -> N,
{
    if lhs.cols() != rhs.shape()[1]
        || lhs.cols() != out.shape()[1]
        || lhs.rows() != rhs.shape()[0]
        || lhs.rows() != out.shape()[0]
    {
        panic!("Dimension mismatch");
    }
    match (
        lhs.storage(),
        rhs.is_standard_layout(),
        out.is_standard_layout(),
    ) {
        (CompressedStorage::CSR, true, true) => (),
        (CompressedStorage::CSC, false, false) => (),
        (_, _, _) => panic!("Storage mismatch"),
    }
    let outer_axis = if rhs.is_standard_layout() {
        Axis(0)
    } else {
        Axis(1)
    };
    for ((mut orow, lrow), rrow) in out
        .axis_iter_mut(outer_axis)
        .zip(lhs.outer_iterator())
        .zip(rhs.axis_iter(outer_axis))
    {
        // now some equivalent of nnz_or_zip is needed
        for items in orow
            .iter_mut()
            .zip(rrow.iter().enumerate().nnz_or_zip(lrow.iter()))
        {
            let (oval, lr_elems) = items;
            let binop_val = match lr_elems {
                Left((_, val)) => binop(val, &N::zero()),
                Right((_, val)) => binop(&N::zero(), val),
                Both((_, lval, rval)) => binop(lval, rval),
            };
            *oval = binop_val;
        }
    }
}

/// Binary operations for CsVec
///
/// This function iterates the non-zero locations of `lhs` and `rhs`
/// and applies the function `binop` to the matching elements (defaulting
/// to zero when e.g. only `lhs` has a non-zero at a given location).
///
/// The function thus has a correct behavior iff `binop(0, 0) == 0`.
pub fn csvec_binop<N, I, F>(
    mut lhs: CsVecViewI<N, I>,
    mut rhs: CsVecViewI<N, I>,
    binop: F,
) -> SpRes<CsVecI<N, I>>
where
    N: Num,
    F: Fn(&N, &N) -> N,
    I: SpIndex,
{
    csvec_fix_zeros(&mut lhs, &mut rhs);
    if lhs.dim() != rhs.dim() {
        panic!("Dimension mismatch");
    }
    let mut res = CsVecI::empty(lhs.dim());
    let max_nnz = lhs.nnz() + rhs.nnz();
    res.reserve_exact(max_nnz);
    for elem in lhs.iter().nnz_or_zip(rhs.iter()) {
        let (ind, binop_val) = match elem {
            Left((ind, val)) => (ind, binop(val, &N::zero())),
            Right((ind, val)) => (ind, binop(&N::zero(), val)),
            Both((ind, lval, rval)) => (ind, binop(lval, rval)),
        };
        res.append(ind, binop_val);
    }
    Ok(res)
}

fn csvec_fix_zeros<N, I: SpIndex>(
    lhs: &mut CsVecViewI<N, I>,
    rhs: &mut CsVecViewI<N, I>,
) {
    if rhs.dim() == 0 {
        rhs.dim = lhs.dim;
    }
    if lhs.dim() == 0 {
        lhs.dim = rhs.dim;
    }
}

#[cfg(test)]
mod test {
    use crate::sparse::CsMat;
    use crate::sparse::CsVec;
    use crate::test_data::{mat1, mat1_times_2, mat2, mat_dense1};
    use ndarray::{arr2, Array};

    fn mat1_plus_mat2() -> CsMat<f64> {
        let indptr = vec![0, 5, 8, 9, 12, 15];
        let indices = vec![0, 1, 2, 3, 4, 0, 3, 4, 2, 1, 2, 3, 1, 2, 3];
        let data =
            vec![6., 7., 6., 4., 3., 8., 11., 5., 5., 8., 2., 4., 4., 4., 7.];
        CsMat::new((5, 5), indptr, indices, data)
    }

    fn mat1_minus_mat2() -> CsMat<f64> {
        let indptr = vec![0, 4, 7, 8, 11, 14];
        let indices = vec![0, 1, 3, 4, 0, 3, 4, 2, 1, 2, 3, 1, 2, 3];
        let data = vec![
            -6., -7., 4., -3., -8., -7., 5., 5., 8., -2., -4., -4., -4., 7.,
        ];
        CsMat::new((5, 5), indptr, indices, data)
    }

    fn mat1_times_mat2() -> CsMat<f64> {
        let indptr = vec![0, 1, 2, 2, 2, 2];
        let indices = vec![2, 3];
        let data = vec![9., 18.];
        CsMat::new((5, 5), indptr, indices, data)
    }

    #[test]
    fn test_add1() {
        let a = mat1();
        let b = mat2();

        let c = super::add_mat_same_storage(&a, &b);
        let c_true = mat1_plus_mat2();
        assert_eq!(c, c_true);

        let c = &a + &b;
        assert_eq!(c, c_true);

        // test with CSR matrices having differ row patterns
        let a = CsMat::new((3, 3), vec![0, 1, 1, 2], vec![0, 2], vec![1., 1.]);
        let b = CsMat::new((3, 3), vec![0, 1, 2, 2], vec![0, 1], vec![1., 1.]);
        let c = CsMat::new(
            (3, 3),
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![2., 1., 1.],
        );

        assert_eq!(c, &a + &b);
    }

    #[test]
    fn test_sub1() {
        let a = mat1();
        let b = mat2();

        let c = super::sub_mat_same_storage(&a, &b);
        let c_true = mat1_minus_mat2();
        assert_eq!(c, c_true);

        let c = &a - &b;
        assert_eq!(c, c_true);
    }

    #[test]
    fn test_mul1() {
        let a = mat1();
        let b = mat2();

        let c = super::mul_mat_same_storage(&a, &b);
        let c_true = mat1_times_mat2();
        assert_eq!(c.indptr(), c_true.indptr());
        assert_eq!(c.indices(), c_true.indices());
        assert_eq!(c.data(), c_true.data());
    }

    #[test]
    fn test_smul() {
        let a = mat1();
        let c = super::scalar_mul_mat(&a, 2.);
        let c_true = mat1_times_2();
        assert_eq!(c.indptr(), c_true.indptr());
        assert_eq!(c.indices(), c_true.indices());
        assert_eq!(c.data(), c_true.data());
    }

    #[test]
    fn csvec_binops() {
        let vec1 = CsVec::new(8, vec![0, 2, 4, 6], vec![1.; 4]);
        let vec2 = CsVec::new(8, vec![1, 3, 5, 7], vec![2.; 4]);
        let vec3 = CsVec::new(8, vec![1, 2, 5, 6], vec![3.; 4]);

        let res = &vec1 + &vec2;
        let expected_output = CsVec::new(
            8,
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![1., 2., 1., 2., 1., 2., 1., 2.],
        );
        assert_eq!(expected_output, res);

        let res = &vec1 + &vec3;
        let expected_output =
            CsVec::new(8, vec![0, 1, 2, 4, 5, 6], vec![1., 3., 4., 1., 3., 4.]);
        assert_eq!(expected_output, res);
    }

    #[test]
    fn zero_sized_vector_works_as_right_vector_operand() {
        let vector = CsVec::new(8, vec![0, 2, 4, 6], vec![1.; 4]);
        let zero = CsVec::<f64>::new(0, vec![], vec![]);
        assert_eq!(&vector + zero, vector);
    }

    #[test]
    fn zero_sized_vector_works_as_left_vector_operand() {
        let vector = CsVec::new(8, vec![0, 2, 4, 6], vec![1.; 4]);
        let zero = CsVec::<f64>::new(0, vec![], vec![]);
        assert_eq!(zero + &vector, vector);
    }

    #[test]
    fn csr_add_dense_rowmaj() {
        let a = Array::zeros((3, 3));
        let b = CsMat::eye(3);

        let c = super::add_dense_mat_same_ordering(&b, &a, 1., 1.);

        let mut expected_output = Array::zeros((3, 3));
        expected_output[[0, 0]] = 1.;
        expected_output[[1, 1]] = 1.;
        expected_output[[2, 2]] = 1.;

        assert_eq!(c, expected_output);

        let a = mat1();
        let b = mat_dense1();

        let expected_output = arr2(&[
            [0., 1., 5., 7., 4.],
            [5., 6., 5., 6., 8.],
            [4., 5., 9., 3., 2.],
            [3., 12., 3., 2., 1.],
            [1., 2., 1., 8., 0.],
        ]);
        let c = super::add_dense_mat_same_ordering(&a, &b, 1., 1.);
        assert_eq!(c, expected_output);
        let c = &a + &b;
        assert_eq!(c, expected_output);
    }

    #[test]
    fn csr_mul_dense_rowmaj() {
        let a = Array::from_elem((3, 3), 1.);
        let b = CsMat::eye(3);

        let c = super::mul_dense_mat_same_ordering(&b, &a, 1.);

        let expected_output = Array::eye(3);

        assert_eq!(c, expected_output);
    }
}
