use sprs::errors::SprsError;
use sprs::{CsStructureI, CsStructureViewI, PermOwnedI, SpIndex};
use suitesparse_camd_sys::*;

/// Find a permutation matrix P which reduces the fill-in of the square
/// sparse matrix `mat` in Cholesky factorization (ie, the number of nonzeros
/// of the Cholesky factorization of P A P^T is less than for the Cholesky
/// factorization of A).
///
/// If A is not symmetric, the ordering will be computed for A + A^T
///
/// # Errors
///
/// This function will error if the passed matrix is not square.
pub fn try_camd<I, Iptr>(
    mat: CsStructureViewI<I, Iptr>,
) -> Result<PermOwnedI<I>, SprsError>
where
    I: SpIndex,
    Iptr: SpIndex,
{
    let n = mat.rows();
    if n != mat.cols() {
        return Err(SprsError::IllegalArguments(
            "Input to camd must be square",
        ));
    }
    let mut perm: Vec<SuiteSparseInt> = vec![0; n];
    let mut control = [0.; CAMD_CONTROL];
    let mut info = [0.; CAMD_INFO];
    let constraint: *const SuiteSparseInt = std::ptr::null();
    // TODO use the long version when required by the size of the matrix
    let mat: CsStructureI<SuiteSparseInt, SuiteSparseInt> =
        mat.to_other_types();
    let res = unsafe {
        camd_order(
            n as SuiteSparseInt,
            mat.indptr().as_ptr(),
            mat.indices().as_ptr(),
            perm.as_mut_ptr(),
            control.as_mut_ptr(),
            info.as_mut_ptr(),
            constraint as *mut SuiteSparseInt,
        )
    };
    // CsMat invariants guarantee sorted and non duplicate indices so this
    // should not happen.
    if res as isize != CAMD_OK {
        return Err(SprsError::NonSortedIndices);
    }
    let perm = perm.iter().map(|&i| I::from_usize(i as usize)).collect();
    Ok(PermOwnedI::new(perm))
}

/// Find a permutation matrix P which reduces the fill-in of the square
/// sparse matrix `mat` in Cholesky factorization (ie, the number of nonzeros
/// of the Cholesky factorization of P A P^T is less than for the Cholesky
/// factorization of A).
///
/// If A is not symmetric, the ordering will be computed for A + A^T
///
/// # Panics
///
/// This function will panic if the passed matrix is not square.
pub fn camd<I, Iptr>(mat: CsStructureViewI<I, Iptr>) -> PermOwnedI<I>
where
    I: SpIndex,
    Iptr: SpIndex,
{
    try_camd(mat).unwrap()
}

#[cfg(test)]
mod tests {
    use sprs::CsMatI;

    #[test]
    fn try_camd() {
        let mat = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1., 2., 21., 6., 6., 2., 2., 8.],
        );
        let res = super::try_camd(mat.structure_view());
        assert!(res.is_ok());
    }
}
