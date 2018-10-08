extern crate num_traits;
extern crate sprs;
extern crate suitesparse_ldl_sys;

use std::ops::Deref;

use num_traits::Num;
use sprs::{CsMatI, CsMatViewI, PermOwnedI, SpIndex};
use suitesparse_ldl_sys::*;

macro_rules! ldl_impl {
    ($int: ty,
     $Symbolic: ident,
     $Numeric: ident,
     $symbolic: ident,
     $numeric: ident,
     $lsolve: ident,
     $dsolve: ident,
     $ltsolve: ident,
     $perm: ident,
     $permt: ident,
     $valid_perm: ident,
     $valid_matrix: ident
     ) => (
        /// Structure holding the symbolic ldlt decomposition computed by
        /// suitesparse's ldl
        #[derive(Debug, Clone)]
        pub struct $Symbolic {
            n: $int,
            lp: Vec<$int>,
            parent: Vec<$int>,
            lnz: Vec<$int>,
            flag: Vec<$int>,
            p: Vec<$int>,
            pinv: Vec<$int>,
        }

        /// Structure holding the numeric ldlt decomposition computed by
        /// suitesparse's ldl
        #[derive(Debug, Clone)]
        pub struct $Numeric {
            symbolic: $Symbolic,
            li: Vec<$int>,
            lx: Vec<f64>,
            d: Vec<f64>,
            y: Vec<f64>,
            pattern: Vec<$int>
        }

        impl $Symbolic {
            /// Compute the symbolic LDLT decomposition of the given matrix.
            ///
            /// # Panics
            ///
            /// * if the matrix is not symmetric
            pub fn new<N, I>(mat: CsMatViewI<N, I>) -> $Symbolic
            where N: Clone + Into<f64>,
                  I: SpIndex,
            {
                assert_eq!(mat.rows(), mat.cols());
                let n = mat.rows();
                $Symbolic::new_perm(mat, PermOwnedI::identity(n))
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
            /// * if perm does not represent a valid permutation
            pub fn new_perm<N, I>(mat: CsMatViewI<N, I>,
                                  perm: PermOwnedI<I>) -> $Symbolic
            where N: Clone + Into<f64>,
                  I: SpIndex,
            {
                assert!(mat.rows() == mat.cols());
                let n = mat.rows();
                let n_ = n as $int;
                let mat: CsMatI<f64, $int> = mat.to_other_types();
                let ap = mat.indptr().as_ptr();
                let ai = mat.indices().as_ptr();
                let valid_mat = unsafe { $valid_matrix(n_, ap, ai) };
                assert!(valid_mat == 1);
                let perm = perm.to_other_idx_type();
                let p = perm.vec();
                let pinv = perm.inv_vec();
                let mut flag = vec![0; n];
                let valid_p = unsafe {
                    $valid_perm(n_, p.as_ptr(), flag.as_mut_ptr())
                };
                let valid_pinv = unsafe {
                    $valid_perm(n_, pinv.as_ptr(), flag.as_mut_ptr())
                };
                assert!(valid_p == 1 && valid_pinv == 1);
                let mut res = $Symbolic {
                    n: n_,
                    lp: vec![0; n + 1],
                    parent: vec![0; n],
                    lnz: vec![0; n],
                    flag: flag,
                    p: p,
                    pinv: pinv,
                };
                unsafe {
                    $symbolic(n_,
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

            /// Factor a matrix, assuming it shares the same nonzero pattern
            /// as the matrix this factorization was built from.
            ///
            /// # Panics
            ///
            /// If the matrix is not symmetric.
            pub fn factor<N, I>(self, mat: CsMatViewI<N, I>) -> $Numeric
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
                let mut ldl_numeric = $Numeric {
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

        impl $Numeric {

            /// Compute the numeric LDLT decomposition of the given matrix.
            ///
            /// # Panics
            ///
            /// * if mat is not symmetric
            pub fn new<N, I>(mat: CsMatViewI<N, I>) -> Self
            where N: Clone + Into<f64>,
                  I: SpIndex,
            {
                let symbolic = $Symbolic::new(mat.view());
                symbolic.factor(mat)
            }

            /// Compute the numeric decomposition L D L^T = P^T A P
            /// where P is a permutation matrix.
            ///
            /// Using a good permutation matrix can reduce the non-zero count in L,
            /// thus making the decomposition and the solves faster.
            ///
            /// # Panics
            ///
            /// * if mat is not symmetric
            pub fn new_perm<N, I>(mat: CsMatViewI<N, I>, perm: PermOwnedI<I>) -> Self
            where N: Clone + Into<f64>,
                  I: SpIndex,
            {
                let symbolic = $Symbolic::new_perm(mat.view(), perm);
                symbolic.factor(mat)
            }

            /// Factor a new matrix, assuming it shares the same nonzero pattern
            /// as the matrix this factorization was built from.
            ///
            /// # Panics
            ///
            /// If the matrix is not symmetric.
            pub fn update<N, I>(&mut self, mat: CsMatViewI<N, I>)
            where N: Clone + Into<f64>,
                  I: SpIndex,
            {
                let mat: CsMatI<f64, $int> = mat.to_other_types();
                let ap = mat.indptr().as_ptr();
                let ai = mat.indices().as_ptr();
                let ax = mat.data().as_ptr();
                assert!(unsafe { $valid_matrix(self.symbolic.n, ap, ai) } != 0);
                unsafe {
                    $numeric(self.symbolic.n,
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
                    $perm(self.symbolic.n,
                          x.as_mut_ptr(),
                          rhs.as_ptr(),
                          self.symbolic.p.as_ptr());
                    $lsolve(self.symbolic.n,
                            x.as_mut_ptr(),
                            self.symbolic.lp.as_ptr(),
                            self.li.as_ptr(),
                            self.lx.as_ptr());
                    $dsolve(self.symbolic.n,
                            x.as_mut_ptr(),
                            self.d.as_ptr());
                    $ltsolve(self.symbolic.n,
                             x.as_mut_ptr(),
                             self.symbolic.lp.as_ptr(),
                             self.li.as_ptr(),
                             self.lx.as_ptr());
                    $permt(self.symbolic.n,
                           y.as_mut_ptr(),
                           x.as_ptr(),
                           self.symbolic.p.as_ptr());
                }
                y.iter().map(|&x| x.into()).collect()
            }

            /// The size of the linear system associated with this decomposition
            #[inline]
            pub fn problem_size(&self) -> usize {
                self.symbolic.problem_size()
            }

            /// The number of non-zero entries in L
            #[inline]
            pub fn nnz(&self) -> usize {
                self.symbolic.nnz()
            }
        }
    )
}

ldl_impl!(
    ldl_int,
    LdlSymbolic,
    LdlNumeric,
    ldl_symbolic,
    ldl_numeric,
    ldl_lsolve,
    ldl_dsolve,
    ldl_ltsolve,
    ldl_perm,
    ldl_permt,
    ldl_valid_perm,
    ldl_valid_matrix
);

ldl_impl!(
    ldl_long,
    LdlLongSymbolic,
    LdlLongNumeric,
    ldl_l_symbolic,
    ldl_l_numeric,
    ldl_l_lsolve,
    ldl_l_dsolve,
    ldl_l_ltsolve,
    ldl_l_perm,
    ldl_l_permt,
    ldl_l_valid_perm,
    ldl_l_valid_matrix
);

#[cfg(test)]
mod tests {
    use super::{LdlLongSymbolic, LdlSymbolic};
    use sprs::{CsMatI, PermOwnedI};

    #[test]
    fn ldl_symbolic() {
        let mat = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1., 2., 21., 6., 6., 2., 2., 8.],
        );
        let perm = PermOwnedI::new(vec![0, 2, 1, 3]);
        let ldlt = LdlSymbolic::new_perm(mat.view(), perm).factor(mat.view());
        let b = vec![9., 60., 18., 34.];
        let x0 = vec![1., 2., 3., 4.];
        let x = ldlt.solve(&b);
        assert_eq!(x, x0);
    }

    #[test]
    fn ldl_long_symbolic() {
        let mat = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1., 2., 21., 6., 6., 2., 2., 8.],
        );
        let perm = PermOwnedI::new(vec![0, 2, 1, 3]);
        let ldlt =
            LdlLongSymbolic::new_perm(mat.view(), perm).factor(mat.view());
        let b = vec![9., 60., 18., 34.];
        let x0 = vec![1., 2., 3., 4.];
        let x = ldlt.solve(&b);
        assert_eq!(x, x0);
    }
}
