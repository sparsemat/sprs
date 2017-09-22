extern crate sprs;
extern crate suitesparse_ldl_sys;

use sprs::{CsMatViewI, CsMatI, SpIndex, PermOwnedI};
use suitesparse_ldl_sys::*;

/// Structure holding the symbolic ldlt decomposition computed by
/// suitesparse's ldl
#[derive(Debug, Clone)]
pub struct LdlSymbolic {
    n: ldl_int,
    lp: Vec<ldl_int>,
    parent: Vec<ldl_int>,
    lnz: Vec<ldl_int>,
    flag: Vec<ldl_int>,
    p: Vec<ldl_int>,
    pinv: Vec<ldl_int>,
}

impl LdlSymbolic {
    /// Compute the symbolic LDLT decomposition of the given matrix.
    ///
    /// # Panics
    ///
    /// * if the matrix is not symmetric
    pub fn new<N, I>(mat: CsMatViewI<N, I>) -> LdlSymbolic
    where N: Clone + Into<f64>,
          I: SpIndex,
    {
        LdlSymbolic::new_perm(mat, PermOwnedI::identity())
    }

    /// Compute the symbolic decomposition L D L^T = P A P^T
    /// where P is a permutation matrix.
    ///
    /// Using a good permutation matrix can reduce the non-zero count in L,
    /// thus making the decomposition and the solves faster.
    ///
    /// # Panics
    ///
    /// * if mat is not symmetric
    pub fn new_perm<N, I>(mat: CsMatViewI<N, I>,
                          perm: PermOwnedI<I>) -> LdlSymbolic
    where N: Clone + Into<f64>,
          I: SpIndex,
    {
        assert!(mat.rows() == mat.cols());
        let n = mat.rows();
        let n_ = n as ldl_int;
        let mat: CsMatI<f64, ldl_int> = mat.to_other_types();
        let ap = mat.indptr().as_ptr() as *mut ldl_int;
        let ai = mat.indices().as_ptr() as *mut ldl_int;
        let valid_mat = unsafe { ldl_valid_matrix(n_, ap, ai) };
        let perm = perm.to_other_idx_type();
        let p = perm.vec(n_);
        let pinv = perm.inv_vec(n_);
        assert!(valid_mat == 1);
        let mut res = LdlSymbolic {
            n: n_,
            lp: vec![0; n + 1],
            parent: vec![0; n],
            lnz: vec![0; n],
            flag: vec![0; n],
            p: p,
            pinv: pinv,
        };
        unsafe {
            ldl_symbolic(n_,
                         ap,
                         ai,
                         res.lp.as_mut_ptr(),
                         res.parent.as_mut_ptr(),
                         res.lnz.as_mut_ptr(),
                         res.flag.as_mut_ptr(),
                         res.p.as_mut_ptr(),
                         res.pinv.as_mut_ptr());
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use sprs::{CsMatI, PermOwnedI};
    use super::LdlSymbolic;

    #[test]
    fn ldl_symbolic() {
        let mat = CsMatI::new_csc((4, 4),
                                  vec![0, 2, 4, 6, 8],
                                  vec![0, 3, 1, 2, 1, 2, 0, 3],
                                  vec![1, 2, 21, 6, 6, 2, 2, 8]);
        let perm = PermOwnedI::new(vec![0, 2, 1, 3]);
        let _ldlt = LdlSymbolic::new_perm(mat.view(), perm);
    }
}
