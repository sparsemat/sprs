//! This module contains functions useful when dealing with linear algebra
//! problems subject to constraints.

use crate::{CsMatI, CsMatViewI, CsVecViewI, PermOwnedI, SpIndex};
use std::fmt::Debug;

type NdVec<N> = ndarray::Array<N, ndarray::Ix1>;
type NdVecView<'a, N> = ndarray::ArrayView<'a, N, ndarray::Ix1>;

/// Compute a linear system corresponding to solving a least-squares problem
/// with simple equality constraints.
///
/// The least-squares problem is the minimization of
/// `(sys_mat * x - rhs).norm().squared()` subject to the constraint
/// `x[i] == v` for all `(i, v)` values in the sparse vector `constraint`.
pub fn eqconst_ls_system<N, I, Iptr>(
    sys_mat: CsMatViewI<N, I, Iptr>,
    rhs: NdVecView<N>,
    constraint: CsVecViewI<N, I>,
) -> (CsMatI<N, I, Iptr>, PermOwnedI<I>, NdVec<N>)
where
    N: num_traits::Zero + Clone + crate::MulAcc + Debug + Send + Sync + Default,
    I: SpIndex,
    Iptr: SpIndex,
{
    // We call A = sys_mat, b = rhs, and x the unkown.
    // We wish to minimize || Ax - b ||^2 under equality constraints. Let P
    // be the matrix that puts the constraint variables on top. Then we want
    // to minimize || AP^Ty - b ||^2 with y = Px.
    // The normal equations are
    // PA^TAP^T y = PA^Tb
    //
    assert!(constraint.dim() <= sys_mat.cols());
    let perm = crate::permute_to_top(constraint.indices(), sys_mat.cols());
    // FIXME: should slice before product for perf
    let apt = sys_mat.view() * &perm.inv();
    let patapt = &apt.transpose_view() * &apt;
    let patb = &apt.transpose_view() * &rhs;
    let ata = &sys_mat.transpose_view() * &sys_mat;
    let ata = crate::transform_mat_papt(ata.view(), perm.view());
    let atb = &perm * (&sys_mat.transpose_view() * &rhs);
    // TODO: need to take only the relevant part, so inner slicing.
    // Alternatively, it should be possible to first apply the permutation
    // matrix to the sys mat, then slice on the outer dim, then compute the
    // normal equation matrix on the reduced matrix. That would also be more
    // performant.
    unimplemented!()
}
