extern crate sprs;
extern crate suitesparse_ldl_sys;
extern crate num_traits;

use std::ops::Deref;

use sprs::{CsMatViewI, CsMatI, SpIndex, PermOwnedI};
use suitesparse_ldl_sys::*;
use num_traits::Num;

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

/// Structure holding the numeric ldlt decomposition computed by
/// suitesparse's ldl
#[derive(Debug, Clone)]
pub struct LdlNumeric {
    symbolic: LdlSymbolic,
    li: Vec<ldl_int>,
    lx: Vec<f64>,
    d: Vec<f64>,
    y: Vec<f64>,
    pattern: Vec<ldl_int>
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
        let ap = mat.indptr().as_ptr();
        let ai = mat.indices().as_ptr();
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
                         res.p.as_ptr(),
                         res.pinv.as_ptr());
        }
        res
    }

    /// The size of the linear system associated with this decomposition
    #[inline]
    pub fn problem_size(&self) -> usize {
        self.n as usize
    }

    /// The number of non-zero entries in L
    #[inline]
    pub fn nnz(&self) -> usize {
        let n = self.problem_size();
        self.lp[n] as usize
    }

    pub fn factor<N, I>(self, mat: CsMatViewI<N, I>) -> LdlNumeric
    where N: Clone + Into<f64>,
          I: SpIndex,
    {
        let n = self.problem_size();
        let nnz = self.nnz();
        let li = vec![0; nnz];
        let lx = vec![0.; nnz];
        let d = vec![0.; n];
        let y = vec![0.; n];
        let pattern = vec![0; n];
        let mut ldl_numeric = LdlNumeric {
            symbolic: self,
            li: li,
            lx: lx,
            d: d,
            y: y,
            pattern: pattern,
        };
        ldl_numeric.update(mat);
        ldl_numeric
    }
}

impl LdlNumeric {

    pub fn update<N, I>(&mut self, mat: CsMatViewI<N, I>)
    where N: Clone + Into<f64>,
          I: SpIndex,
    {
        let mat: CsMatI<f64, ldl_int> = mat.to_other_types();
        let ap = mat.indptr().as_ptr();
        let ai = mat.indices().as_ptr();
        let ax = mat.data().as_ptr();
        unsafe {
            ldl_numeric(self.symbolic.n,
                        ap,
                        ai,
                        ax,
                        self.symbolic.lp.as_mut_ptr(),
                        self.symbolic.parent.as_mut_ptr(),
                        self.symbolic.lnz.as_mut_ptr(),
                        self.li.as_mut_ptr(),
                        self.lx.as_mut_ptr(),
                        self.d.as_mut_ptr(),
                        self.y.as_mut_ptr(),
                        self.pattern.as_mut_ptr(),
                        self.symbolic.flag.as_mut_ptr(),
                        self.symbolic.p.as_ptr(),
                        self.symbolic.pinv.as_ptr());
        }
    }

    /// Solve the system A x = rhs
    pub fn solve<'a, N, V>(&self, rhs: &V) -> Vec<N>
    where N: 'a + Copy + Num + Into<f64> + From<f64>,
          V: Deref<Target = [N]>
    {
        assert!(self.symbolic.n as usize == rhs.len());
        let mut x = vec![0.; rhs.len()];
        let mut y = x.clone();
        let rhs: Vec<f64> = rhs.iter().map(|&x| x.into()).collect();
        unsafe {
            ldl_perm(self.symbolic.n,
                     x.as_mut_ptr(),
                     rhs.as_ptr(),
                     self.symbolic.p.as_ptr());
            ldl_lsolve(self.symbolic.n,
                       x.as_mut_ptr(),
                       self.symbolic.lp.as_ptr(),
                       self.li.as_ptr(),
                       self.lx.as_ptr());
            ldl_dsolve(self.symbolic.n,
                       x.as_mut_ptr(),
                       self.d.as_ptr());
            ldl_ltsolve(self.symbolic.n,
                        x.as_mut_ptr(),
                        self.symbolic.lp.as_ptr(),
                        self.li.as_ptr(),
                        self.lx.as_ptr());
            ldl_permt(self.symbolic.n,
                      y.as_mut_ptr(),
                      x.as_ptr(),
                      self.symbolic.p.as_ptr());
        }
        y.iter().map(|&x| x.into()).collect()
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
                                  vec![1., 2., 21., 6., 6., 2., 2., 8.]);
        let perm = PermOwnedI::new(vec![0, 2, 1, 3]);
        let ldlt = LdlSymbolic::new_perm(mat.view(), perm)
            .factor(mat.view());
        let b = vec![9., 60., 18., 34.];
        let x0 = vec![1., 2., 3., 4.];
        let x = ldlt.solve(&b);
        assert_eq!(x, x0);
    }
}
