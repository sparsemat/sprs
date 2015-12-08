/// Cholesky factorization

use std::ops::Deref;
use std::ops::IndexMut;

use num::traits::Num;

use sparse::csmat::{self, CsMat, CsMatView};
use sparse::symmetric::is_symmetric;
use sparse::permutation::{Permutation, PermOwned};
use utils::csmat_borrowed_uchk;
use sparse::linalg::{self, etree};
use stack::DStack;

pub enum SymmetryCheck {
    CheckSymmetry,
    DontCheckSymmetry,
}

/// Structure to compute a  symbolic LDLT decomposition
#[derive(Debug)]
pub struct LdlSymbolic {
    colptr: Vec<usize>,
    parents: linalg::etree::ParentsOwned,
    nz: Vec<usize>,
    flag_workspace: Vec<usize>,
    perm: Permutation<Vec<usize>>,
}

#[derive(Debug)]
pub struct LdlNumeric<N> {
    symbolic: LdlSymbolic,
    l_indices: Vec<usize>,
    l_data: Vec<N>,
    diag: Vec<N>,
    y_workspace: Vec<N>,
    pattern_workspace: DStack<usize>,
}

impl LdlSymbolic {
    pub fn new<N, IpS, IS, DS>(mat: &CsMat<N, IpS, IS, DS>) -> LdlSymbolic
    where N: Copy + PartialEq,
          IpS: Deref<Target = [usize]>,
          IS: Deref<Target = [usize]>,
          DS: Deref<Target = [N]>
    {
        let perm: Permutation<Vec<usize>> = Permutation::identity();
        LdlSymbolic::new_perm(mat, perm)
    }

    pub fn new_perm<N, IpS, IS, DS>(mat: &CsMat<N, IpS, IS, DS>,
                                    perm: PermOwned)
                                    -> LdlSymbolic
    where N: Copy + PartialEq,
          IpS: Deref<Target = [usize]>,
          IS: Deref<Target = [usize]>,
          DS: Deref<Target = [N]>
    {
        let n = mat.cols();
        assert!(mat.rows() == n, "matrix should be square");
        let mut l_colptr = vec![0; n+1];
        let mut parents = linalg::etree::ParentsOwned::new(n);
        let mut l_nz = vec![0; n];
        let mut flag_workspace = vec![0; n];
        ldl_symbolic(mat.borrowed(),
                     &perm,
                     &mut l_colptr,
                     parents.borrowed_mut(),
                     &mut l_nz,
                     &mut flag_workspace,
                     SymmetryCheck::CheckSymmetry);

        LdlSymbolic {
            colptr: l_colptr,
            parents: parents,
            nz: l_nz,
            flag_workspace: flag_workspace,
            perm: perm,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.parents.nb_nodes()
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        let n = self.dim();
        self.colptr[n]
    }

    pub fn factor<N, IpS, IS, DS>(self,
                                  mat: &CsMat<N, IpS, IS, DS>)
                                  -> LdlNumeric<N>
    where N: Copy + Num + PartialOrd,
          IpS: Deref<Target = [usize]>,
          IS: Deref<Target = [usize]>,
          DS: Deref<Target = [N]>
    {
        let n = self.dim();
        let nnz = self.nnz();
        let l_indices = vec![0; nnz];
        let l_data = vec![N::zero(); nnz];
        let diag = vec![N::zero(); n];
        let y_workspace = vec![N::zero(); n];
        let pattern_workspace = DStack::with_capacity(n);
        let mut ldl_numeric = LdlNumeric {
            symbolic: self,
            l_indices: l_indices,
            l_data: l_data,
            diag: diag,
            y_workspace: y_workspace,
            pattern_workspace: pattern_workspace,
        };
        ldl_numeric.update(mat);
        ldl_numeric
    }
}

impl<N> LdlNumeric<N> {

    pub fn new<IpS, IS, DS>(mat: &CsMat<N, IpS, IS, DS>) -> Self
    where N: Copy + Num + PartialOrd,
          IpS: Deref<Target = [usize]>,
          IS: Deref<Target = [usize]>,
          DS: Deref<Target = [N]>
    {
        let symbolic = LdlSymbolic::new(mat);
        symbolic.factor(mat)
    }

    pub fn new_perm<IpS, IS, DS>(mat: &CsMat<N, IpS, IS, DS>,
                                 perm: PermOwned)
                                 -> Self
    where N: Copy + Num + PartialOrd,
          IpS: Deref<Target = [usize]>,
          IS: Deref<Target = [usize]>,
          DS: Deref<Target = [N]>
    {
        let symbolic = LdlSymbolic::new_perm(mat, perm);
        symbolic.factor(mat)
    }

    pub fn update<IpS, IS, DS>(&mut self, mat: &CsMat<N, IpS, IS, DS>)
    where N: Copy + Num + PartialOrd,
          IpS: Deref<Target = [usize]>,
          IS: Deref<Target = [usize]>,
          DS: Deref<Target = [N]>
    {
        ldl_numeric(mat.borrowed(),
                    &self.symbolic.colptr,
                    self.symbolic.parents.borrowed(),
                    &self.symbolic.perm,
                    &mut self.symbolic.nz,
                    &mut self.l_indices,
                    &mut self.l_data,
                    &mut self.diag,
                    &mut self.y_workspace,
                    &mut self.pattern_workspace,
                    &mut self.symbolic.flag_workspace);
    }

    pub fn solve<'a, V>(&self, rhs: &V) -> Vec<N>
    where N: 'a + Copy + Num,
          V: Deref<Target = [N]>
    {
        let mut x = &self.symbolic.perm * &rhs[..];
        let n = self.symbolic.dim();
        let l = csmat_borrowed_uchk(csmat::CSC,
                                    n,
                                    n,
                                    &self.symbolic.colptr,
                                    &self.l_indices,
                                    &self.l_data);
        ldl_lsolve(&l, &mut x);
        linalg::diag_solve(&self.diag, &mut x);
        ldl_ltsolve(&l, &mut x);
        let pinv = self.symbolic.perm.inv();
        &pinv * &x
    }
}

/// Perform a symbolic LDLt decomposition of a symmetric sparse matrix
pub fn ldl_symbolic<N, PStorage>(mat: CsMatView<N>,
                                 perm: &Permutation<PStorage>,
                                 l_colptr: &mut [usize],
                                 mut parents: etree::ParentsViewMut,
                                 l_nz: &mut [usize],
                                 flag_workspace: &mut [usize],
                                 check_symmetry: SymmetryCheck)
where N: Clone + Copy + PartialEq,
      PStorage: Deref<Target = [usize]>
{

    match check_symmetry {
        SymmetryCheck::DontCheckSymmetry => (),
        SymmetryCheck::CheckSymmetry => if !is_symmetric(&mat) {
            panic!("Matrix is not symmetric")
        },
    }

    let n = mat.rows();

    let outer_it = mat.outer_iterator_perm(perm.borrowed());
    // compute the elimination tree of L
    for (k, (_, vec)) in outer_it.enumerate() {

        flag_workspace[k] = k; // this node is visited
        parents.set_root(k);
        l_nz[k] = 0;

        for (inner_ind, _) in vec.iter_perm(perm.inv()) {
            let mut i = inner_ind;

            if i < k {
                while flag_workspace[i] != k {
                    parents.uproot(i, k);
                    l_nz[i] += 1;
                    flag_workspace[i] = k;
                    i = parents.get_parent(i).expect("uprooted so not a root");
                }
            }
        }
    }

    let mut prev: usize = 0;
    for (k, colptr) in (0..n).zip(l_colptr.iter_mut()) {
        *colptr = prev;
        prev += l_nz[k];
    }
    l_colptr[n] = prev;

}

/// Perform numeric LDLT decomposition
///
/// pattern_workspace is a DStack of capacity n
pub fn ldl_numeric<N, PStorage>(mat: CsMatView<N>,
                                l_colptr: &[usize],
                                parents: etree::ParentsView,
                                perm: &Permutation<PStorage>,
                                l_nz: &mut [usize],
                                l_indices: &mut [usize],
                                l_data: &mut [N],
                                diag: &mut [N],
                                y_workspace: &mut [N],
                                pattern_workspace: &mut DStack<usize>,
                                flag_workspace: &mut [usize])
where N: Clone + Copy + PartialEq + Num + PartialOrd,
      PStorage: Deref<Target = [usize]>
{
    let outer_it = mat.outer_iterator_perm(perm.borrowed());
    for (k, (_, vec)) in outer_it.enumerate() {

        // compute the nonzero pattern of the kth row of L
        // in topological order

        flag_workspace[k] = k; // this node is visited
        y_workspace[k] = N::zero();
        l_nz[k] = 0;
        pattern_workspace.clear_right();

        for (inner_ind, val) in vec.iter_perm(perm.inv())
                                   .filter(|&(i, _)| i <= k) {
            y_workspace[inner_ind] = y_workspace[inner_ind] + val;
            let mut i = inner_ind;
            pattern_workspace.clear_left();
            while flag_workspace[i] != k {
                pattern_workspace.push_left(i);
                flag_workspace[i] = k;
                i = parents.get_parent(i).expect("enforced by ldl_symbolic");
            }
            pattern_workspace.push_left_on_right();
        }

        // use a sparse triangular solve to compute the values
        // of the kth row of L
        diag[k] = y_workspace[k];
        y_workspace[k] = N::zero();
        'pattern: for &i in pattern_workspace.iter_right() {
            let yi = y_workspace[i];
            y_workspace[i] = N::zero();
            let p2 = l_colptr[i] + l_nz[i];
            for p in l_colptr[i]..p2 {
                // we cannot go inside this loop before something has actually
                // be written into l_indices[l_colptr[i]..p2] so this
                // read is actually not into garbage
                // actually each iteration of the 'pattern loop adds writes the
                // value in l_indices that will be read on the next iteration
                // TODO: can some design change make this fact more obvious?
                let y_index = l_indices[p];
                y_workspace[y_index] = y_workspace[y_index] - l_data[p] * yi;
            }
            let l_ki = yi / diag[i];
            diag[k] = diag[k] - l_ki * yi;
            l_indices[p2] = k;
            l_data[p2] = l_ki;
            l_nz[i] += 1;
        }
        if diag[k] == N::zero() {
            panic!("Matrix is singular");
        }
    }
}

/// Triangular solve specialized on lower triangular matrices
/// produced by ldlt (diagonal terms are omitted and assumed to be 1).
pub fn ldl_lsolve<N, V: ?Sized>(l: &CsMatView<N>, x: &mut V)
where N: Clone + Copy + Num,
      V: IndexMut<usize, Output = N>
{
    for (col_ind, vec) in l.outer_iterator() {
        let x_col = x[col_ind];
        for (row_ind, value) in vec.iter() {
            x[row_ind] = x[row_ind] - value * x_col;
        }
    }
}

/// Triangular transposed solve specialized on lower triangular matrices
/// produced by ldlt (diagonal terms are omitted and assumed to be 1).
pub fn ldl_ltsolve<N, V: ?Sized>(l: &CsMatView<N>, x: &mut V)
where N: Clone + Copy + Num,
      V: IndexMut<usize, Output = N>
{
    for (outer_ind, vec) in l.outer_iterator().rev() {
        let mut x_outer = x[outer_ind];
        for (inner_ind, value) in vec.iter() {
            x_outer = x_outer - value * x[inner_ind];
        }
        x[outer_ind] = x_outer;
    }
}

#[cfg(test)]
mod test {
    use sparse::csmat::{self, CsMat, CsMatOwned};
    use sparse::csmat::CompressedStorage::CSC;
    use sparse::permutation::Permutation;
    use sparse::linalg;
    use super::SymmetryCheck;
    use utils::csmat_borrowed_uchk;
    use stack::DStack;

    fn test_mat1() -> CsMatOwned<f64> {
        let indptr = vec![0, 2, 5, 6, 7, 13, 14, 17, 20, 24, 28];
        let indices = vec![0, 8, 1, 4, 9, 2, 3, 1, 4, 6, 7, 8, 9, 5, 4, 6, 9,
                           4, 7, 8, 0, 4, 7, 8, 1, 4, 6, 9];
        let data = vec![1.7, 0.13, 1., 0.02, 0.01, 1.5, 1.1, 0.02, 2.6, 0.16,
                        0.09, 0.52, 0.53, 1.2, 0.16, 1.3, 0.56, 0.09, 1.6,
                        0.11, 0.13, 0.52, 0.11, 1.4, 0.01, 0.53, 0.56, 3.1];
        CsMat::new_owned(CSC, 10, 10, indptr, indices, data).unwrap()
    }

    fn test_vec1() -> Vec<f64> {
        vec![0.287, 0.22, 0.45, 0.44, 2.486, 0.72, 1.55, 1.424, 1.621, 3.759]
    }

    fn expected_factors1() -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
        let expected_lp = vec![0, 1, 3, 3, 3, 7, 7, 10, 12, 13, 13];
        let expected_li = vec![8, 4, 9, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9];
        let expected_lx = vec![0.076470588235294124,
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
                               -0.071852278207562709];
        let expected_d = vec![1.7,
                              1.,
                              1.5,
                              1.1000000000000001,
                              2.5996000000000001,
                              1.2,
                              1.290152331127866,
                              1.5968603527854308,
                              1.2799646117414738,
                              2.7695677698030283];
        (expected_lp, expected_li, expected_lx, expected_d)
    }

    fn expected_lsolve_res1() -> Vec<f64> {
        vec![0.28699999999999998,
             0.22,
             0.45000000000000001,
             0.44,
             2.4816000000000003,
             0.71999999999999997,
             1.3972626557931991,
             1.3440844395148306,
             1.0599997771886431,
             2.7695677698030279]
    }

    fn expected_dsolve_res1() -> Vec<f64> {
        vec![0.16882352941176471,
             0.22,
             0.29999999999999999,
             0.39999999999999997,
             0.95460840129250657,
             0.59999999999999998,
             1.0830214557467768,
             0.84170443406044937,
             0.82814772179243734,
             0.99999999999999989]
    }

    fn expected_res1() -> Vec<f64> {
        vec![0.099999999999999992,
             0.19999999999999998,
             0.29999999999999999,
             0.39999999999999997,
             0.5,
             0.59999999999999998,
             0.70000000000000007,
             0.79999999999999993,
             0.90000000000000002,
             0.99999999999999989]
    }

    #[test]
    fn test_factor1() {
        let mut l_colptr = [0; 11];
        let mut parents = linalg::etree::ParentsOwned::new(10);
        let mut l_nz = [0; 10];
        let mut flag_workspace = [0; 10];
        let perm: Permutation<&[usize]> = Permutation::identity();
        let mat = test_mat1();
        super::ldl_symbolic(mat.borrowed(),
                            &perm,
                            &mut l_colptr,
                            parents.borrowed_mut(),
                            &mut l_nz,
                            &mut flag_workspace,
                            SymmetryCheck::CheckSymmetry);

        let nnz = l_colptr[10];
        let mut l_indices = vec![0; nnz];
        let mut l_data = vec![0.; nnz];
        let mut diag = [0.; 10];
        let mut y_workspace = [0.; 10];
        let mut pattern_workspace = DStack::with_capacity(10);
        super::ldl_numeric(mat.borrowed(),
                           &l_colptr,
                           parents.borrowed(),
                           &perm,
                           &mut l_nz,
                           &mut l_indices,
                           &mut l_data,
                           &mut diag,
                           &mut y_workspace,
                           &mut pattern_workspace,
                           &mut flag_workspace);

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
        let l = csmat_borrowed_uchk(csmat::CSC,
                                    n,
                                    n,
                                    &expected_lp,
                                    &expected_li,
                                    &expected_lx);
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
        let ldlt = super::LdlNumeric::new(&mat);
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

        let mat = CsMatOwned::new_owned(CSC,
                                        4,
                                        4,
                                        vec![0, 2, 4, 6, 8],
                                        vec![0, 3, 1, 2, 1, 2, 0, 3],
                                        vec![1, 2, 21, 6, 6, 2, 2, 8])
                      .unwrap();

        let perm = Permutation::new(vec![0, 2, 1, 3]);

        let ldlt = super::LdlNumeric::new_perm(&mat, perm);
        let b = vec![9, 60, 18, 34];
        let x0 = vec![1, 2, 3, 4];
        let x = ldlt.solve(&b);
        assert_eq!(x, x0);
    }
}
