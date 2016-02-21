///! Sparse matrix addition, subtraction

use sparse::csmat::{CsMat, CsMatOwned, CsMatView, CompressedStorage};
use num::traits::Num;
use sparse::vec::NnzEither::{Left, Right, Both};
use sparse::vec::{CsVec, CsVecView, CsVecOwned, SparseIterTools};
use sparse::compressed::SpMatView;
use errors::SprsError;
use ndarray::{self, OwnedArray, ArrayBase, ArrayView, ArrayViewMut, Ix};

/// Sparse matrix addition, with matrices sharing the same storage type
pub fn add_mat_same_storage<N, Mat1, Mat2>(
    lhs: &Mat1, rhs: &Mat2) -> Result<CsMatOwned<N>, SprsError>
where N: Num + Copy, Mat1: SpMatView<N>, Mat2: SpMatView<N> {
    csmat_binop(lhs.borrowed(), rhs.borrowed(), |&x, &y| x + y)
}

/// Sparse matrix subtraction, with same storage type
pub fn sub_mat_same_storage<N, Mat1, Mat2>(
    lhs: &Mat1, rhs: &Mat2) -> Result<CsMatOwned<N>, SprsError>
where N: Num + Copy, Mat1: SpMatView<N>, Mat2: SpMatView<N> {
    csmat_binop(lhs.borrowed(), rhs.borrowed(), |&x, &y| x - y)
}

/// Sparse matrix scalar multiplication, with same storage type
pub fn mul_mat_same_storage<N, Mat1, Mat2>(
    lhs: &Mat1, rhs: &Mat2) -> Result<CsMatOwned<N>, SprsError>
where N: Num + Copy, Mat1: SpMatView<N>, Mat2: SpMatView<N> {
    csmat_binop(lhs.borrowed(), rhs.borrowed(), |&x, &y| x * y)
}

/// Sparse matrix multiplication by a scalar
pub fn scalar_mul_mat<N, Mat>(
    mat: &Mat, val: N) -> CsMatOwned<N>
where N: Num + Copy, Mat: SpMatView<N> {
    let mat = mat.borrowed();
    let mut out_indptr = vec![0; mat.outer_dims() + 1];
    let mut out_indices = vec![0; mat.nb_nonzero()];
    let mut out_data = vec![N::zero(); mat.nb_nonzero()];
    let nrows = mat.rows();
    let ncols = mat.cols();
    let storage_type = mat.storage();
    scalar_mul_mat_raw(mat, val, &mut out_indptr[..],
                       &mut out_indices[..], &mut out_data[..]);
    CsMat::new_owned(storage_type, nrows, ncols,
                     out_indptr, out_indices, out_data).unwrap()
}

/// Sparse matrix multiplication by a scalar, raw version
///
/// Writes into the provided output.
/// Panics if the sizes don't match
pub fn scalar_mul_mat_raw<N>(
    mat: CsMatView<N>,
    val: N,
    out_indptr: &mut [usize],
    out_indices: &mut [usize],
    out_data: &mut [N])
where N: Num + Copy {
    assert_eq!(out_indptr.len(), mat.outer_dims() + 1);
    assert!(out_data.len() >= mat.nb_nonzero());
    assert!(out_indices.len() >= mat.nb_nonzero());
    for (optr, iptr) in out_indptr.iter_mut().zip(mat.indptr()) {
        *optr = *iptr;
    }
    for (oind, iind) in out_indices.iter_mut().zip(mat.indices()) {
        *oind = *iind;
    }
    for (odata, idata) in out_data.iter_mut().zip(mat.data()) {
        *odata = *idata * val;
    }
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
pub fn csmat_binop<N, F>(lhs: CsMatView<N>,
                         rhs: CsMatView<N>,
                         binop: F
                        ) -> Result<CsMatOwned<N>, SprsError>
where N: Num,
      F: Fn(&N, &N) -> N
{
    let nrows = lhs.rows();
    let ncols = lhs.cols();
    let storage_type = lhs.storage();
    if nrows != rhs.rows() || ncols != rhs.cols() {
        return Err(SprsError::IncompatibleDimensions);
    }
    if storage_type != rhs.storage() {
        return Err(SprsError::IncompatibleStorages);
    }

    let max_nnz = lhs.nb_nonzero() + rhs.nb_nonzero();
    let mut out_indptr = vec![0; lhs.outer_dims() + 1];
    let mut out_indices = vec![0; max_nnz];

    // Sadly the vec! macro requires Clone, but we don't want to force
    // Clone on our consumers, so we have to use this workaround.
    // This should compile to decent code however.
    let mut out_data = Vec::with_capacity(max_nnz);
    for _ in 0..max_nnz {
        out_data.push(N::zero());
    }

    let nnz = csmat_binop_same_storage_raw(lhs, rhs, binop,
                                           &mut out_indptr[..],
                                           &mut out_indices[..],
                                           &mut out_data[..]);
    out_indices.truncate(nnz);
    out_data.truncate(nnz);
    Ok(CsMat::new_owned(storage_type, nrows, ncols,
                        out_indptr, out_indices, out_data).unwrap())
}


/// Raw implementation of scalar binary operation for compressed sparse matrices
/// sharing the same storage. The output arrays are assumed to be preallocated
///
/// Returns the nnz count
pub fn csmat_binop_same_storage_raw<N, F>(lhs: CsMatView<N>,
                                          rhs: CsMatView<N>,
                                          binop: F,
                                          out_indptr: &mut [usize],
                                          out_indices: &mut [usize],
                                          out_data: &mut [N]
                                         ) -> usize
where N: Num,
      F: Fn(&N, &N) -> N
{
    assert_eq!(lhs.cols(), rhs.cols());
    assert_eq!(lhs.rows(), rhs.rows());
    assert_eq!(lhs.storage(), rhs.storage());
    assert_eq!(out_indptr.len(), rhs.outer_dims() + 1);
    let max_nnz = lhs.nb_nonzero() + rhs.nb_nonzero();
    assert!(out_data.len() >= max_nnz);
    assert!(out_indices.len() >= max_nnz);
    let mut nnz = 0;
    out_indptr[0] = 0;
    for ((dim, lv), (_, rv)) in lhs.outer_iterator().zip(rhs.outer_iterator()) {
        for elem in lv.iter().nnz_or_zip(rv.iter()) {
            let (ind, binop_val) = match elem {
                Left((ind, val)) => (ind, binop(val, &N::zero())),
                Right((ind, val)) => (ind, binop(&N::zero(), val)),
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

/// Compute alpha * lhs + beta * rhs with lhs a sparse matrix and rhs dense
/// and alpha and beta scalars
pub fn add_dense_mat_same_ordering<N, Mat, DenseStorage>(
    lhs: &Mat,
    rhs: &ArrayBase<DenseStorage, (Ix, Ix)>,
    alpha: N,
    beta: N)
-> Result<OwnedArray<N, (Ix, Ix)>, SprsError>
where N: Num + Copy,
      Mat: SpMatView<N>,
      DenseStorage: ndarray::Data<Elem=N> {
    let binop = |x, y| alpha * x + beta * y;
    let shape = (rhs.shape()[0], rhs.shape()[1]);
    let mut res = match rhs.is_standard_layout() {
        true => OwnedArray::zeros(shape),
        false => OwnedArray::zeros_f(shape),
    };
    try!(csmat_binop_dense_same_ordering_raw(lhs.borrowed(),
                                             rhs.view(),
                                             binop,
                                             res.view_mut()));
    Ok(res)
}

/// Compute coeff wise alpha * lhs * rhs with lhs a sparse matrix and rhs dense
/// and alpha a scalar
pub fn mul_dense_mat_same_ordering<N, Mat, DenseStorage>(
    lhs: &Mat, rhs: &ArrayBase<DenseStorage, (Ix, Ix)>,
    alpha: N)
-> Result<OwnedArray<N, (Ix, Ix)>, SprsError>
where N: Num + Copy, Mat: SpMatView<N>, DenseStorage: ndarray::Data<Elem=N> {
    let binop = |x, y| alpha * x * y;
    let shape = (rhs.shape()[0], rhs.shape()[1]);
    let mut res = match rhs.is_standard_layout() {
        true => OwnedArray::zeros(shape),
        false => OwnedArray::zeros_f(shape),
    };
    try!(csmat_binop_dense_same_ordering_raw(lhs.borrowed(),
                                             rhs.view(),
                                             binop,
                                             res.view_mut()));
    Ok(res)
}


/// Raw implementation of sparse/dense binary operations with the same
/// ordering
pub fn csmat_binop_dense_same_ordering_raw<'a, N, F>(lhs: CsMatView<'a, N>,
                                                     rhs: ArrayView<'a, N, (Ix, Ix)>,
                                                     binop: F,
                                                     mut out: ArrayViewMut<'a, N, (Ix, Ix)>
                                                    ) -> Result<(), SprsError>
where N: 'a + Copy + Num,
      F: Fn(N, N) -> N {
    if         lhs.cols() != rhs.shape()[1] || lhs.cols() != out.shape()[1]
            || lhs.rows() != rhs.shape()[0] || lhs.rows() != out.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    match (lhs.storage(), rhs.is_standard_layout(), out.is_standard_layout()) {
        (CompressedStorage::CSR, true, true) => (),
        (CompressedStorage::CSC, false, false) => (),
        (_, _, _) => return Err(SprsError::IncompatibleStorages),
    }
    //let outer_axis = tensor::Axis(rhs.outer_dim().unwrap());
    let outer_axis = if rhs.is_standard_layout() { 0 } else { 1 };
    for ((mut orow, (_, lrow)), rrow) in out.axis_iter_mut(outer_axis)
                                            .zip(lhs.outer_iterator())
                                            .zip(rhs.axis_iter(outer_axis)) {
        // now some equivalent of nnz_or_zip is needed
        for items in orow.iter_mut()
                         .zip(rrow.iter().enumerate().nnz_or_zip(lrow.iter())) {
            let (mut oval, lr_elems) = items;
            let binop_val = match lr_elems {
                Left((_, &val)) => binop(val, N::zero()),
                Right((_, &val)) => binop(N::zero(), val),
                Both((_, &lval, &rval)) => binop(lval, rval),
            };
            *oval = binop_val;
        }
    }
    Ok(())
}

/// Binary operations for CsVec
///
/// This function iterates the non-zero locations of `lhs` and `rhs`
/// and applies the function `binop` to the matching elements (defaulting
/// to zero when e.g. only `lhs` has a non-zero at a given location).
///
/// The function thus has a correct behavior iff `binop(0, 0) == 0`.
pub fn csvec_binop<N, F>(lhs: CsVecView<N>,
                         rhs: CsVecView<N>,
                         binop: F
                        ) -> Result<CsVecOwned<N>, SprsError>
where N: Num,
      F: Fn(&N, &N) -> N
{
    if lhs.dim() != rhs.dim() {
        return Err(SprsError::IncompatibleDimensions);
    }
    let mut res = CsVec::empty(lhs.dim());
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

#[cfg(test)]
mod test {
    use sparse::csmat::{CsMat, CsMatOwned};
    use sparse::vec::CsVec;
    use sparse::CompressedStorage::{CSR};
    use test_data::{mat1, mat2, mat1_times_2, mat_dense1};
    use ndarray::{arr2, OwnedArray};

    fn mat1_plus_mat2() -> CsMatOwned<f64> {
        let indptr = vec![0,  5,  8,  9, 12, 15];
        let indices = vec![0, 1, 2, 3, 4, 0, 3, 4, 2, 1, 2, 3, 1, 2, 3];
        let data = vec![6.,  7.,  6.,  4.,  3.,
                        8.,  11.,  5.,  5.,  8.,
                        2.,  4.,  4.,  4.,  7.];
        CsMat::new_owned(CSR, 5, 5, indptr, indices, data).unwrap()
    }

    fn mat1_minus_mat2() -> CsMatOwned<f64> {
        let indptr = vec![0,  4,  7,  8, 11, 14];
        let indices = vec![0, 1, 3, 4, 0, 3, 4, 2, 1, 2, 3, 1, 2, 3];
        let data = vec![-6., -7.,  4., -3., -8.,
                        -7.,  5.,  5.,  8., -2.,
                        -4., -4., -4.,  7.];
        CsMat::new_owned(CSR, 5, 5, indptr, indices, data).unwrap()
    }

    fn mat1_times_mat2() -> CsMatOwned<f64> {
        let indptr = vec![0,  1,  2,  2, 2, 2];
        let indices = vec![2, 3];
        let data = vec![9., 18.];
        CsMat::new_owned(CSR, 5, 5, indptr, indices, data).unwrap()
    }


    #[test]
    fn test_add1() {
        let a = mat1();
        let b = mat2();

        let c = super::add_mat_same_storage(&a, &b).unwrap();
        let c_true = mat1_plus_mat2();
        assert_eq!(c, c_true);

        let c = &a + &b;
        assert_eq!(c, c_true);

        // test with CSR matrices having differ row patterns
        let a = CsMatOwned::new_owned(CSR, 3, 3,
                                      vec![0, 1, 1, 2],
                                      vec![0, 2],
                                      vec![1., 1.]).unwrap();
        let b = CsMatOwned::new_owned(CSR, 3, 3,
                                      vec![0, 1, 2, 2],
                                      vec![0, 1],
                                      vec![1., 1.]).unwrap();
        let c = CsMatOwned::new_owned(CSR, 3, 3,
                                      vec![0, 1, 2, 3],
                                      vec![0, 1, 2],
                                      vec![2., 1., 1.]).unwrap();

        assert_eq!(c, &a + &b);
    }

    #[test]
    fn test_sub1() {
        let a = mat1();
        let b = mat2();

        let c = super::sub_mat_same_storage(&a, &b).unwrap();
        let c_true = mat1_minus_mat2();
        assert_eq!(c, c_true);

        let c = &a - &b;
        assert_eq!(c, c_true);
    }

    #[test]
    fn test_mul1() {
        let a = mat1();
        let b = mat2();

        let c = super::mul_mat_same_storage(&a, &b).unwrap();
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
        let vec1 = CsVec::new_owned(8, vec![0, 2, 4, 6], vec![1.; 4]).unwrap();
        let vec2 = CsVec::new_owned(8, vec![1, 3, 5, 7], vec![2.; 4]).unwrap();
        let vec3 = CsVec::new_owned(8, vec![1, 2, 5, 6], vec![3.; 4]).unwrap();

        let res = &vec1 + &vec2;
        let expected_output = CsVec::new_owned(
            8, vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![1., 2., 1., 2., 1., 2., 1., 2.]).unwrap();
        assert_eq!(expected_output, res);

        let res = &vec1 + &vec3;
        let expected_output = CsVec::new_owned(8,
                                               vec![0, 1, 2, 4, 5, 6],
                                               vec![1., 3., 4., 1., 3., 4.]
                                              ).unwrap();
        assert_eq!(expected_output, res);
    }

    #[test]
    fn csr_add_dense_rowmaj() {
        let a = OwnedArray::zeros((3,3));
        let b = CsMatOwned::eye(CSR, 3);

        let c = super::add_dense_mat_same_ordering(&b, &a, 1., 1.).unwrap();

        let mut expected_output = OwnedArray::zeros((3,3));
        expected_output[[0,0]] = 1.;
        expected_output[[1,1]] = 1.;
        expected_output[[2,2]] = 1.;

        assert_eq!(c, expected_output);

        let a = mat1();
        let b = mat_dense1();

        let expected_output = arr2(&[[0., 1., 5., 7., 4.],
                                     [5., 6., 5., 6., 8.],
                                     [4., 5., 9., 3., 2.],
                                     [3., 12., 3., 2., 1.],
                                     [1., 2., 1., 8., 0.]]);
        let c = super::add_dense_mat_same_ordering(&a, &b, 1., 1.).unwrap();
        assert_eq!(c, expected_output);
        let c = &a + &b;
        assert_eq!(c, expected_output);
    }

    #[test]
    fn csr_mul_dense_rowmaj() {
        let a = OwnedArray::from_elem((3,3), 1.);
        let b = CsMatOwned::eye(CSR, 3);

        let c = super::mul_dense_mat_same_ordering(&b, &a, 1.).unwrap();

        let expected_output = OwnedArray::eye(3);

        assert_eq!(c, expected_output);
    }

}
