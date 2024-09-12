//! Cholesky factorization module.
//!
//! Contains LDLT decomposition methods.
//!
//! This decomposition operates on symmetric positive definite matrices,
//! and is written `A = L D L` where L is lower triangular and D is diagonal.
//! It is closely related to the Cholesky decomposition, but is often more
//! numerically stable and can work on some indefinite matrices.
//!
//! The easiest way to use this API is to create a `LdlNumeric` instance from
//! a matrix, then use the `LdlNumeric::solve` method.
//!
//! It is possible to update a decomposition if the sparsity structure of a
//! matrix does not change. In that case the `LdlNumeric::update` method can
//! be used.
//!
//! When only the sparsity structure of a matrix is known, it is possible
//! to precompute part of the factorization by using the `LdlSymbolic` struct.
//! This struct can the be converted into a `LdlNumeric` once the non-zero
//! values are known, using the `LdlSymbolic::factor` method.
//!
//! This method is adapted from the LDL library by Tim Davis:
//!
//! LDL Copyright (c) 2005 by Timothy A. Davis.  All Rights Reserved.
//!
//! LDL License:
//!
//! Your use or distribution of LDL or any modified version of
//! LDL implies that you agree to this License.
//!
//! This library is free software; you can redistribute it and/or
//! modify it under the terms of the GNU Lesser General Public
//! License as published by the Free Software Foundation; either
//! version 2.1 of the License, or (at your option) any later version.
//!
//! This library is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//! Lesser General Public License for more details.
//!
//! You should have received a copy of the GNU Lesser General Public
//! License along with this library; if not, write to the Free Software
//! Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301
//! USA
//!
//! Permission is hereby granted to use or copy this program under the
//! terms of the GNU LGPL, provided that the Copyright, this License,
//! and the Availability of the original version is retained on all copies.
//! User documentation of any code that uses this code or any modified
//! version of this code must cite the Copyright, this License, the
//! Availability note, and "Used by permission." Permission to modify
//! the code and to distribute modified code is granted, provided the
//! Copyright, this License, and the Availability note are retained,
//! and a notice that the code was modified is included.
use std::ops::Deref;

use num_traits::Num;

use sprs::errors::{LinalgError, SingularMatrixInfo};
use sprs::indexing::SpIndex;
use sprs::linalg;
use sprs::stack::DStack;
use sprs::{is_symmetric, CsMatViewI, PermOwnedI, Permutation};
use sprs::{DenseVector, DenseVectorMut};
use sprs::{FillInReduction, PermutationCheck, SymmetryCheck};

#[cfg(feature = "sprs_suitesparse_ldl")]
use sprs_suitesparse_ldl::LdlNumeric as LdlNumericC;
#[cfg(feature = "sprs_suitesparse_ldl")]
use sprs_suitesparse_ldl::LdlSymbolic as LdlSymbolicC;
#[cfg(feature = "sprs_suitesparse_ldl")]
use sprs_suitesparse_ldl::{LdlLongNumeric, LdlLongSymbolic};

/// Builder pattern structure to customize a LDLT decomposition
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Ldl {
    check_symmetry: SymmetryCheck,
    check_perm: PermutationCheck,
    fill_red_method: FillInReduction,
}

impl Default for Ldl {
    fn default() -> Self {
        Self {
            check_symmetry: SymmetryCheck::CheckSymmetry,
            fill_red_method: FillInReduction::ReverseCuthillMcKee,
            check_perm: PermutationCheck::CheckPerm,
        }
    }
}

/// Structure to compute and hold a symbolic LDLT decomposition
#[derive(Debug, Clone)]
pub struct LdlSymbolic<I> {
    colptr: Vec<I>,
    parents: linalg::etree::ParentsOwned,
    nz: Vec<I>,
    flag_workspace: Vec<I>,
    perm: Permutation<I, Vec<I>>,
}

/// Structure to hold a numeric LDLT decomposition
#[derive(Debug, Clone)]
pub struct LdlNumeric<N, I> {
    symbolic: LdlSymbolic<I>,
    l_indices: Vec<I>,
    l_data: Vec<N>,
    diag: Vec<N>,
    y_workspace: Vec<N>,
    pattern_workspace: DStack<I>,
}

impl Ldl {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn check_symmetry(self, check: SymmetryCheck) -> Self {
        Self {
            check_symmetry: check,
            ..self
        }
    }

    pub fn check_perm(self, check: PermutationCheck) -> Self {
        Self {
            check_perm: check,
            ..self
        }
    }

    pub fn fill_in_reduction(self, method: FillInReduction) -> Self {
        Self {
            fill_red_method: method,
            ..self
        }
    }

    pub fn perm<N, I>(&self, mat: CsMatViewI<N, I>) -> PermOwnedI<I>
    where
        I: SpIndex,
    {
        match self.fill_red_method {
            FillInReduction::NoReduction => PermOwnedI::identity(mat.rows()),
            FillInReduction::ReverseCuthillMcKee => {
                sprs::linalg::reverse_cuthill_mckee(mat.structure_view()).perm
            }
            FillInReduction::CAMDSuiteSparse => {
                #[cfg(not(feature = "sprs_suitesparse_camd"))]
                panic!(
                    "Unavailable without the `sprs_suitesparse_camd` feature"
                );
                #[cfg(feature = "sprs_suitesparse_camd")]
                sprs_suitesparse_camd::camd(mat.structure_view())
            }
            _ => {
                unreachable!(
                    "Unhandled method, report a bug at https://github.com/vbarrielle/sprs/issues/199"
                )
            }
        }
    }

    pub fn symbolic<N, I>(self, mat: CsMatViewI<N, I>) -> LdlSymbolic<I>
    where
        I: SpIndex,
        N: Copy + PartialEq,
    {
        LdlSymbolic::new_perm(mat, self.perm(mat), self.check_symmetry)
    }

    #[cfg(feature = "sprs_suitesparse_ldl")]
    pub fn symbolic_c<N, I>(self, mat: CsMatViewI<N, I>) -> LdlSymbolicC
    where
        I: SpIndex,
        N: Copy + PartialEq + Into<f64>,
    {
        LdlSymbolicC::new_perm(mat, self.perm(mat), self.check_perm)
    }

    #[cfg(feature = "sprs_suitesparse_ldl")]
    pub fn symbolic_c_long<N, I>(self, mat: CsMatViewI<N, I>) -> LdlLongSymbolic
    where
        I: SpIndex,
        N: Copy + PartialEq + Into<f64>,
    {
        LdlLongSymbolic::new_perm(mat, self.perm(mat), self.check_perm)
    }

    pub fn numeric<N, I>(
        self,
        mat: CsMatViewI<N, I>,
    ) -> Result<LdlNumeric<N, I>, LinalgError>
    where
        I: SpIndex,
        N: Copy + Num + PartialOrd,
    {
        // self.symbolic(mat).factor(mat)
        let symb = self.symbolic(mat);
        symb.factor(mat)
    }

    #[cfg(feature = "sprs_suitesparse_ldl")]
    pub fn numeric_c<N, I>(
        self,
        mat: CsMatViewI<N, I>,
    ) -> Result<LdlNumericC, LinalgError>
    where
        I: SpIndex,
        N: Copy + Num + PartialOrd + Into<f64>,
    {
        self.symbolic_c(mat).factor(mat)
    }

    #[cfg(feature = "sprs_suitesparse_ldl")]
    pub fn numeric_c_long<N, I>(
        self,
        mat: CsMatViewI<N, I>,
    ) -> Result<LdlLongNumeric, LinalgError>
    where
        I: SpIndex,
        N: Copy + Num + PartialOrd + Into<f64>,
    {
        self.symbolic_c_long(mat).factor(mat)
    }
}

impl<I: SpIndex> LdlSymbolic<I> {
    /// Compute the symbolic LDLT of the given matrix
    ///
    /// # Panics
    ///
    /// * if mat is not symmetric
    pub fn new<N>(mat: CsMatViewI<N, I>) -> Self
    where
        N: Copy + PartialEq,
    {
        assert_eq!(mat.rows(), mat.cols());
        let perm: Permutation<I, Vec<I>> = Permutation::identity(mat.rows());
        Self::new_perm(mat, perm, SymmetryCheck::CheckSymmetry)
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
    pub fn new_perm<N>(
        mat: CsMatViewI<N, I>,
        perm: PermOwnedI<I>,
        check_symmetry: SymmetryCheck,
    ) -> Self
    where
        N: Copy + PartialEq,
        I: SpIndex,
    {
        let n = mat.cols();
        assert!(mat.rows() == n, "matrix should be square");
        let mut l_colptr = vec![I::zero(); n + 1];
        let mut parents = linalg::etree::ParentsOwned::new(n);
        let mut l_nz = vec![I::zero(); n];
        let mut flag_workspace = vec![I::zero(); n];
        ldl_symbolic(
            mat,
            &perm,
            &mut l_colptr,
            parents.view_mut(),
            &mut l_nz,
            &mut flag_workspace,
            check_symmetry,
        );

        Self {
            colptr: l_colptr,
            parents,
            nz: l_nz,
            flag_workspace,
            perm,
        }
    }

    /// The size of the linear system associated with this decomposition
    #[inline]
    pub fn problem_size(&self) -> usize {
        self.parents.nb_nodes()
    }

    /// The number of non-zero entries in L
    #[inline]
    pub fn nnz(&self) -> usize {
        let n = self.problem_size();
        self.colptr[n].index()
    }

    /// Compute the numerical decomposition of the given matrix.
    pub fn factor<N>(
        self,
        mat: CsMatViewI<N, I>,
    ) -> Result<LdlNumeric<N, I>, LinalgError>
    where
        N: Copy + Num + PartialOrd,
    {
        let n = self.problem_size();
        let nnz = self.nnz();
        let l_indices = vec![I::zero(); nnz];
        let l_data = vec![N::zero(); nnz];
        let diag = vec![N::zero(); n];
        let y_workspace = vec![N::zero(); n];
        let pattern_workspace = DStack::with_capacity(n);
        let mut ldl_numeric = LdlNumeric {
            symbolic: self,
            l_indices,
            l_data,
            diag,
            y_workspace,
            pattern_workspace,
        };
        ldl_numeric.update(mat).map(|_| ldl_numeric)
    }
}

impl<N, I: SpIndex> LdlNumeric<N, I> {
    /// Compute the numeric LDLT decomposition of the given matrix.
    ///
    /// # Panics
    ///
    /// * if mat is not symmetric
    pub fn new(mat: CsMatViewI<N, I>) -> Result<Self, LinalgError>
    where
        N: Copy + Num + PartialOrd,
    {
        let symbolic = LdlSymbolic::new(mat.view());
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
    pub fn new_perm(
        mat: CsMatViewI<N, I>,
        perm: PermOwnedI<I>,
        check_symmetry: SymmetryCheck,
    ) -> Result<Self, LinalgError>
    where
        N: Copy + Num + PartialOrd,
    {
        let symbolic = LdlSymbolic::new_perm(mat.view(), perm, check_symmetry);
        symbolic.factor(mat)
    }

    /// Update the decomposition with the given matrix. The matrix must
    /// have the same non-zero pattern as the original matrix, otherwise
    /// the result is unspecified.
    pub fn update(&mut self, mat: CsMatViewI<N, I>) -> Result<(), LinalgError>
    where
        N: Copy + Num + PartialOrd,
    {
        ldl_numeric(
            mat.view(),
            &self.symbolic.colptr,
            self.symbolic.parents.view(),
            &self.symbolic.perm,
            &mut self.symbolic.nz,
            &mut self.l_indices,
            &mut self.l_data,
            &mut self.diag,
            &mut self.y_workspace,
            &mut self.pattern_workspace,
            &mut self.symbolic.flag_workspace,
        )
    }

    /// Solve the system A x = rhs
    ///
    /// The type constraints look complicated, but they simply mean that
    /// `rhs` should be interpretable as a dense vector, and we will return
    /// a dense vector of a compatible type (but owned).
    pub fn solve<'a, V>(
        &self,
        rhs: V,
    ) -> <<V as DenseVector>::Owned as DenseVector>::Owned
    where
        N: 'a + Copy + Num + std::ops::SubAssign + std::ops::DivAssign,
        N: for<'r> std::ops::DivAssign<&'r N>,
        V: DenseVector<Scalar = N>,
        <V as DenseVector>::Owned: DenseVectorMut + DenseVector<Scalar = N>,
        for<'b> &'b <V as DenseVector>::Owned: DenseVector<Scalar = N>,
        for<'b> &'b mut <V as DenseVector>::Owned:
            DenseVectorMut + DenseVector<Scalar = N>,
        <<V as DenseVector>::Owned as DenseVector>::Owned:
            DenseVectorMut + DenseVector<Scalar = N>,
    {
        let mut x = &self.symbolic.perm * rhs;
        let l = self.l();
        ldl_lsolve(&l, &mut x);
        linalg::diag_solve(&self.diag, &mut x);
        ldl_ltsolve(&l, &mut x);
        let pinv = self.symbolic.perm.inv();
        &pinv * x
    }

    /// The diagonal factor D of the LDL^T decomposition
    pub fn d(&self) -> &[N] {
        &self.diag[..]
    }

    /// The L factor of the LDL^T decomposition
    pub fn l(&self) -> CsMatViewI<N, I> {
        use std::slice::from_raw_parts;
        let n = self.symbolic.problem_size();
        // CsMat invariants are guaranteed by the LDL algorithm
        unsafe {
            let indptr = from_raw_parts(self.symbolic.colptr.as_ptr(), n + 1);
            let nnz = indptr[n].index();
            let indices = from_raw_parts(self.l_indices.as_ptr(), nnz);
            let data = from_raw_parts(self.l_data.as_ptr(), nnz);
            CsMatViewI::new_unchecked(sprs::CSC, (n, n), indptr, indices, data)
        }
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

/// Perform a symbolic LDLT decomposition of a symmetric sparse matrix
pub fn ldl_symbolic<N, I, PStorage>(
    mat: CsMatViewI<N, I>,
    perm: &Permutation<I, PStorage>,
    l_colptr: &mut [I],
    mut parents: linalg::etree::ParentsViewMut,
    l_nz: &mut [I],
    flag_workspace: &mut [I],
    check_symmetry: SymmetryCheck,
) where
    N: Clone + Copy + PartialEq,
    I: SpIndex,
    PStorage: Deref<Target = [I]>,
{
    match check_symmetry {
        SymmetryCheck::DontCheckSymmetry => (),
        SymmetryCheck::CheckSymmetry => {
            if !is_symmetric(&mat) {
                panic!("Matrix is not symmetric")
            }
        }
    }

    let n = mat.rows();

    let outer_it = mat.outer_iterator_papt(perm.view());
    // compute the elimination tree of L
    for (k, (_, vec)) in outer_it.enumerate() {
        flag_workspace[k] = I::from_usize(k); // this node is visited
        parents.set_root(k);
        l_nz[k] = I::zero();

        for (inner_ind, _) in vec.iter_perm(perm.inv()) {
            let mut i = inner_ind;

            if i < k {
                while flag_workspace[i].index() != k {
                    parents.uproot(i, k);
                    l_nz[i] += I::one();
                    flag_workspace[i] = I::from_usize(k);
                    i = parents.get_parent(i).expect("uprooted so not a root");
                }
            }
        }
    }

    let mut prev = I::zero();
    for (k, colptr) in (0..n).zip(l_colptr.iter_mut()) {
        *colptr = prev;
        prev += l_nz[k];
    }
    l_colptr[n] = prev;
}

/// Perform numeric LDLT decomposition
///
/// `pattern_workspace` is a [`DStack`] of capacity n
#[allow(clippy::too_many_arguments)]
pub fn ldl_numeric<N, I, PStorage>(
    mat: CsMatViewI<N, I>,
    l_colptr: &[I],
    parents: linalg::etree::ParentsView,
    perm: &Permutation<I, PStorage>,
    l_nz: &mut [I],
    l_indices: &mut [I],
    l_data: &mut [N],
    diag: &mut [N],
    y_workspace: &mut [N],
    pattern_workspace: &mut DStack<I>,
    flag_workspace: &mut [I],
) -> Result<(), LinalgError>
where
    N: Clone + Copy + PartialEq + Num + PartialOrd,
    I: SpIndex,
    PStorage: Deref<Target = [I]>,
{
    assert!(y_workspace.len() == mat.outer_dims());
    assert!(diag.len() == mat.outer_dims());
    let outer_it = mat.outer_iterator_papt(perm.view());
    for (k, (_, vec)) in outer_it.enumerate() {
        // compute the nonzero pattern of the kth row of L
        // in topological order

        flag_workspace[k] = I::from_usize(k); // this node is visited
        y_workspace[k] = N::zero();
        l_nz[k] = I::zero();
        pattern_workspace.clear_right();

        for (inner_ind, &val) in
            vec.iter_perm(perm.inv()).filter(|&(i, _)| i <= k)
        {
            y_workspace[inner_ind] = y_workspace[inner_ind] + val;
            let mut i = inner_ind;
            pattern_workspace.clear_left();
            while flag_workspace[i].index_unchecked() != k {
                pattern_workspace.push_left(I::from_usize(i));
                flag_workspace[i] = I::from_usize(k);
                i = parents.get_parent(i).expect("enforced by ldl_symbolic");
            }
            pattern_workspace.push_left_on_right();
        }

        // use a sparse triangular solve to compute the values
        // of the kth row of L
        diag[k] = y_workspace[k];
        y_workspace[k] = N::zero();
        #[allow(unused_labels)]
        'pattern: for &i in pattern_workspace.iter_right() {
            let i = i.index_unchecked();
            let yi = y_workspace[i];
            y_workspace[i] = N::zero();
            let p2 = (l_colptr[i] + l_nz[i]).index();
            let r0 = l_colptr[i].index()..p2;
            let r1 = l_colptr[i].index()..p2; // Hack because not Copy
            for (y_index, lx) in l_indices[r0].iter().zip(l_data[r1].iter()) {
                // we cannot go inside this loop before something has actually
                // be written into l_indices[l_colptr[i]..p2] so this
                // read is actually not into garbage
                // actually each iteration of the 'pattern loop adds writes the
                // value in l_indices that will be read on the next iteration
                // TODO: can some design change make this fact more obvious?
                // This means we always know it will fit in an usize
                let y_index = y_index.index_unchecked();
                // Safety: `y_index` can take the values taken by `k`, so
                // it is in `0..mat.outer_dims()`, and we have asserted
                // that `y_workspace.len() == mat.outer_dims()`.
                unsafe {
                    let yw = y_workspace.get_unchecked_mut(y_index);
                    *yw = *yw - *lx * yi;
                }
            }
            // Safety: i and k are <= `mat.outer_dims()` and we have asserted
            // that `diag.len() == mat.outer_dims()`.
            let di = *unsafe { diag.get_unchecked(i) };
            let dk = unsafe { diag.get_unchecked_mut(k) };
            let l_ki = yi / di;
            *dk = *dk - l_ki * yi;
            l_indices[p2] = I::from_usize(k);
            l_data[p2] = l_ki;
            l_nz[i] += I::one();
        }
        if diag[k] == N::zero() {
            return Err(LinalgError::SingularMatrix(SingularMatrixInfo {
                index: k,
                reason: "diagonal element is a numeric 0",
            }));
        }
    }
    Ok(())
}

/// Triangular solve specialized on lower triangular matrices
/// produced by ldlt (diagonal terms are omitted and assumed to be 1).
pub fn ldl_lsolve<N, I, V>(l: &CsMatViewI<N, I>, mut x: V)
where
    N: Clone + Copy + Num + std::ops::SubAssign,
    I: SpIndex,
    V: DenseVectorMut + DenseVector<Scalar = N>,
{
    for (col_ind, vec) in l.outer_iterator().enumerate() {
        let x_col = *x.index(col_ind);
        for (row_ind, &value) in vec.iter() {
            *x.index_mut(row_ind) -= value * x_col;
        }
    }
}

/// Triangular transposed solve specialized on lower triangular matrices
/// produced by ldlt (diagonal terms are omitted and assumed to be 1).
pub fn ldl_ltsolve<N, I, V>(l: &CsMatViewI<N, I>, mut x: V)
where
    N: Clone + Copy + Num + std::ops::SubAssign,
    I: SpIndex,
    V: DenseVectorMut + DenseVector<Scalar = N>,
{
    for (outer_ind, vec) in l.outer_iterator().enumerate().rev() {
        let mut x_outer = *x.index(outer_ind);
        for (inner_ind, &value) in vec.iter() {
            x_outer -= value * *x.index(inner_ind);
        }
        *x.index_mut(outer_ind) = x_outer;
    }
}

#[cfg(test)]
mod test {
    use super::SymmetryCheck;
    use sprs::stack::DStack;
    use sprs::{self, linalg, CsMat, CsMatView, Permutation};

    fn test_mat1() -> CsMat<f64> {
        let indptr = vec![0, 2, 5, 6, 7, 13, 14, 17, 20, 24, 28];
        let indices = vec![
            0, 8, 1, 4, 9, 2, 3, 1, 4, 6, 7, 8, 9, 5, 4, 6, 9, 4, 7, 8, 0, 4,
            7, 8, 1, 4, 6, 9,
        ];
        let data = vec![
            1.7, 0.13, 1., 0.02, 0.01, 1.5, 1.1, 0.02, 2.6, 0.16, 0.09, 0.52,
            0.53, 1.2, 0.16, 1.3, 0.56, 0.09, 1.6, 0.11, 0.13, 0.52, 0.11, 1.4,
            0.01, 0.53, 0.56, 3.1,
        ];
        CsMat::new_csc((10, 10), indptr, indices, data)
    }

    fn test_vec1() -> Vec<f64> {
        vec![
            0.287, 0.22, 0.45, 0.44, 2.486, 0.72, 1.55, 1.424, 1.621, 3.759,
        ]
    }

    fn expected_factors1() -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
        let expected_lp = vec![0, 1, 3, 3, 3, 7, 7, 10, 12, 13, 13];
        let expected_li = vec![8, 4, 9, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9];
        let expected_lx = vec![
            0.076470588235294124,
            0.02,
            0.01,
            0.061547930450838589,
            0.034620710878596701,
            0.20003077396522542,
            0.20380058470533929,
            -0.0042935346524025902,
            -0.024807089102770519,
            0.40878266366119237,
            0.05752526570865537,
            -0.010068305077340346,
            -0.071852278207562709,
        ];
        let expected_d = vec![
            1.7,
            1.,
            1.5,
            1.1000000000000001,
            2.5996000000000001,
            1.2,
            1.290152331127866,
            1.5968603527854308,
            1.2799646117414738,
            2.7695677698030283,
        ];
        (expected_lp, expected_li, expected_lx, expected_d)
    }

    fn expected_lsolve_res1() -> Vec<f64> {
        vec![
            0.28699999999999998,
            0.22,
            0.45000000000000001,
            0.44,
            2.4816000000000003,
            0.71999999999999997,
            1.3972626557931991,
            1.3440844395148306,
            1.0599997771886431,
            2.7695677698030279,
        ]
    }

    fn expected_dsolve_res1() -> Vec<f64> {
        vec![
            0.16882352941176471,
            0.22,
            0.29999999999999999,
            0.39999999999999997,
            0.95460840129250657,
            0.59999999999999998,
            1.0830214557467768,
            0.84170443406044937,
            0.82814772179243734,
            0.99999999999999989,
        ]
    }

    fn expected_res1() -> Vec<f64> {
        vec![
            0.099999999999999992,
            0.19999999999999998,
            0.29999999999999999,
            0.39999999999999997,
            0.5,
            0.59999999999999998,
            0.70000000000000007,
            0.79999999999999993,
            0.90000000000000002,
            0.99999999999999989,
        ]
    }

    #[test]
    fn test_factor1() {
        let mut l_colptr = [0; 11];
        let mut parents = linalg::etree::ParentsOwned::new(10);
        let mut l_nz = [0; 10];
        let mut flag_workspace = [0; 10];
        let perm: Permutation<usize, &[usize]> = Permutation::identity(10);
        let mat = test_mat1();
        super::ldl_symbolic(
            mat.view(),
            &perm,
            &mut l_colptr,
            parents.view_mut(),
            &mut l_nz,
            &mut flag_workspace,
            SymmetryCheck::CheckSymmetry,
        );

        let nnz = l_colptr[10];
        let mut l_indices = vec![0; nnz];
        let mut l_data = vec![0.; nnz];
        let mut diag = [0.; 10];
        let mut y_workspace = [0.; 10];
        let mut pattern_workspace = DStack::with_capacity(10);
        super::ldl_numeric(
            mat.view(),
            &l_colptr,
            parents.view(),
            &perm,
            &mut l_nz,
            &mut l_indices,
            &mut l_data,
            &mut diag,
            &mut y_workspace,
            &mut pattern_workspace,
            &mut flag_workspace,
        )
        .unwrap();

        let (expected_lp, expected_li, expected_lx, expected_d) =
            expected_factors1();

        assert_eq!(&l_colptr, &expected_lp[..]);
        assert_eq!(&l_indices, &expected_li);
        assert_eq!(&l_data, &expected_lx);
        assert_eq!(&diag, &expected_d[..]);
    }

    #[test]
    fn test_solve1() {
        let (expected_lp, expected_li, expected_lx, expected_d) =
            expected_factors1();
        let b = test_vec1();
        let mut x = b.clone();
        let n = b.len();
        let l = CsMatView::new_csc(
            (n, n),
            &expected_lp,
            &expected_li,
            &expected_lx,
        );
        super::ldl_lsolve(&l, &mut x);
        assert_eq!(&x, &expected_lsolve_res1());
        linalg::diag_solve(&expected_d, &mut x);
        assert_eq!(&x, &expected_dsolve_res1());
        super::ldl_ltsolve(&l, &mut x);

        let x0 = expected_res1();
        assert_eq!(x, x0);
    }

    #[test]
    fn test_factor_solve1() {
        let mat = test_mat1();
        let b = test_vec1();
        let ldlt = super::LdlNumeric::new(mat.view()).unwrap();
        let x = ldlt.solve(&b);
        let x0 = expected_res1();
        assert_eq!(x, x0);
    }

    #[test]
    fn permuted_ldl_solve() {
        // |1      | |1      | |1     2|   |1      | |1      2| |1      |
        // |  1    | |  2    | |  1 3  |   |    1  | |  21 6  | |    1  |
        // |  3 1  | |    3  | |    1  | = |  1    | |   6 2  | |  1    |
        // |2     1| |      4| |      1|   |      1| |2      8| |      1|
        //     L         D        L^T    =     P          A        P^T
        //
        // |1      2| |1|   | 9|
        // |  21 6  | |2|   |60|
        // |   6 2  | |3| = |18|
        // |2      8| |4|   |34|

        let mat = CsMat::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1, 2, 21, 6, 6, 2, 2, 8],
        );

        let perm = Permutation::new(vec![0, 2, 1, 3]);

        let ldlt = super::LdlNumeric::new_perm(
            mat.view(),
            perm,
            super::SymmetryCheck::CheckSymmetry,
        )
        .unwrap();
        let b = vec![9, 60, 18, 34];
        let x0 = vec![1, 2, 3, 4];
        let x = ldlt.solve(&b);
        assert_eq!(x, x0);
    }

    #[test]
    fn cuthill_ldl_solve() {
        let mat = CsMat::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1., 2., 21., 6., 6., 2., 2., 8.],
        );

        let b = ndarray::arr1(&[9., 60., 18., 34.]);
        let x0 = ndarray::arr1(&[1., 2., 3., 4.]);

        let ldlt = super::Ldl::new()
            .check_symmetry(super::SymmetryCheck::DontCheckSymmetry)
            .fill_in_reduction(super::FillInReduction::ReverseCuthillMcKee)
            .numeric(mat.view())
            .unwrap();
        let x = ldlt.solve(b.view());
        assert_eq!(x, x0);
    }

    #[cfg(feature = "sprs_suitesparse_ldl")]
    #[test]
    fn cuthill_ldl_solve_c() {
        let mat = CsMat::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1., 2., 21., 6., 6., 2., 2., 8.],
        );

        let b = vec![9., 60., 18., 34.];
        let x0 = vec![1., 2., 3., 4.];

        let ldlt = super::Ldl::new()
            .check_perm(super::PermutationCheck::CheckPerm)
            .fill_in_reduction(super::FillInReduction::ReverseCuthillMcKee)
            .numeric_c(mat.view())
            .unwrap();
        let x = ldlt.solve(&b);
        assert_eq!(x, x0);
    }

    #[cfg(feature = "sprs_suitesparse_camd")]
    #[test]
    fn camd_ldl_solve() {
        // 0 - A - 2 - 3
        // | \ | \ | / |
        // 7 - 5 - 6 - 4
        // | / | / | \ |
        // 8 - 9 - 1 - E
        #[rustfmt::skip]
        let triangles = ndarray::arr2(
            &[[0, 7, 5],
              [0, 5, 10],
              [10, 5, 6],
              [10, 6, 2],
              [2, 6, 3],
              [3, 6, 4],
              [7, 8, 5],
              [5, 8, 9],
              [5, 9, 6],
              [6, 9, 1],
              [6, 1, 11],
              [6, 11, 4]],
        );
        let lap_mat =
            sprs::special_mats::tri_mesh_graph_laplacian(12, triangles.view());
        let ldlt_camd = super::Ldl::new()
            .check_symmetry(super::SymmetryCheck::DontCheckSymmetry)
            .fill_in_reduction(super::FillInReduction::CAMDSuiteSparse)
            .numeric(lap_mat.view())
            .unwrap();
        let ldlt_cuthill = super::Ldl::new()
            .check_symmetry(super::SymmetryCheck::DontCheckSymmetry)
            .fill_in_reduction(super::FillInReduction::ReverseCuthillMcKee)
            .numeric(lap_mat.view())
            .unwrap();
        let ldlt_raw = super::Ldl::new()
            .check_symmetry(super::SymmetryCheck::DontCheckSymmetry)
            .fill_in_reduction(super::FillInReduction::NoReduction)
            .numeric(lap_mat.view())
            .unwrap();
        assert!(ldlt_camd.nnz() < ldlt_raw.nnz());
        assert!(ldlt_camd.nnz() < ldlt_cuthill.nnz());
    }
}
