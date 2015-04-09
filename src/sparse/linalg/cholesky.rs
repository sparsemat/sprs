/// Cholesky factorization

use std::ops::{Deref};

use num::traits::Num;

use sparse::csmat::{CsMat, CompressedStorage};
use sparse::symmetric::{is_symmetric};
use sparse::permutation::Permutation;

/// Result of the symbolic LDLt decomposition of a symmetric sparse matrix
pub struct SymbolicLDL {
    /// Dimension of the matrix
    n : usize,
    /// indptr in the L matrix, len is n+1
    l_colptr : Vec<usize>,
    /// Elimination tree, len is n
    parents : Vec<isize>,
    /// number of nonzeros in each column of L, len is n
    l_nz : Vec<usize>,
    /// permutation matrix
    perm : Permutation<Vec<usize>>
}

pub enum SymmetryCheck {
    CheckSymmetry,
    DontCheckSymmetry
}

/// Perform a symbolic LDLt decomposition of a symmetric sparse matrix
pub fn ldl_symbolic<N, IStorage, DStorage, PStorage>(
    mat: &CsMat<N, IStorage, DStorage>,
    perm: &Permutation<PStorage>,
    l_colptr: &mut [usize],
    parents: &mut [isize],
    l_nz: &mut [usize],
    flag_workspace: &mut [usize],
    check_symmetry: SymmetryCheck) -> Result<(),()>
where
N: Clone + Copy + PartialEq,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]>,
PStorage: Deref<Target=[usize]> {

    match check_symmetry {
        SymmetryCheck::DontCheckSymmetry => (),
        SymmetryCheck::CheckSymmetry => if ! is_symmetric(mat) {
            return Err(());
        }
    }

    let n = mat.rows();

    let mut parents = vec![-1isize; n];
    let mut l_nz = vec![0usize; n];

    for (k, (outer_ind, vec)) in mat.outer_iterator_papt(&perm.borrowed()).enumerate() {

        flag_workspace[k] = k; // this node is visited

        for (inner_ind, _) in vec.iter() {
            let mut i = inner_ind;

            // FIXME: the article tests inner_ind versus k, but this looks
            // weird as it would introduce a dissimetry between the permuted
            // and non permuted cases. Needs test however
            if i < outer_ind {
                // get back to the root of the etree
                // TODO: maybe this calls for a more adequate parent structure?
                while flag_workspace[i] != outer_ind {
                    if parents[i] == -1 {
                        parents[i] = outer_ind as isize; // TODO check overflow
                    }
                    l_nz[i] = l_nz[i] + 1;
                    flag_workspace[i] = outer_ind;
                    i = parents[i] as usize; // TODO check negative
                }
            }
        }
    }

    let mut prev : usize = 0;
    for (k, colptr) in (0..n).zip(l_colptr.iter_mut()) {
        *colptr = prev;
        prev += l_nz[k];
    }
    l_colptr[n-1] = prev;

    Ok(())
}

pub fn ldl_numeric<N, IStorage, DStorage, PStorage>(
    n: usize,
    mat: &CsMat<N, IStorage, DStorage>,
    l_colptr: &[usize],
    parents: &[isize],
    perm: &Permutation<PStorage>,
    l_nz: &mut [usize],
    l_indices: &mut [usize],
    l_data: &mut [N],
    diag: &mut [N],
    y_workspace: &mut [N],
    pattern_workspace: &mut [usize],
    flag_workspace: &mut [usize]) -> Result<(), usize>
where
N: Clone + Copy + PartialEq + Num + PartialOrd,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]>,
PStorage: Deref<Target=[usize]> {

    for (k, (outer_ind, vec))
    in mat.outer_iterator_papt(&perm.borrowed()).enumerate() {

        // compute the nonzero pattern of the kth row of L
        // in topological order

        flag_workspace[k] = k; // this node is visited
        y_workspace[k] = N::zero();
        l_nz[k] = 0;
        let mut top = n;

        for (inner_ind, val) in vec.iter().filter(|&(i,_)| i <= k) {
            y_workspace[inner_ind] = y_workspace[inner_ind] + val;
            let mut i = inner_ind;
            //let mut len = 0;
            while flag_workspace[i] != outer_ind {
                top -= 1;
                //pattern_workspace[len] = i;
                pattern_workspace[top] = i;
                //len += 1;
                flag_workspace[i] = k;
                i = parents[i] as usize;
            }
            //while len > 0 { // TODO: can be written as a loop with iterators
            //    top -= 1;
            //    len -= 1;
            //    pattern_workspace[top] = pattern_workspace[len];
            //}
        }

        // use a sparse triangular solve to compute the values
        // of the kth row of L
        diag[k] = y_workspace[k];
        y_workspace[k] = N::zero();
        'pattern: for &i in &pattern_workspace[top..n] {
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
            return Err(k)
        }
    }
    Ok(())
}

pub fn ldl_lsolve<N>(
    l_colptr: &[usize],
    l_indices: &[usize],
    l_data: &[N],
    x: &mut [N])
where
N: Clone + Copy + Num {

    let n = l_indices.len();
    let l = CsMat::from_slices(
        CompressedStorage::CSC, n, n, l_colptr, l_indices, l_data).unwrap();
    for (col_ind, vec) in l.outer_iterator() {
        for (row_ind, value) in vec.iter() {
            x[col_ind] = x[col_ind] - value * x[row_ind];
        }
    }
}

pub fn ldl_ltsolve<N>(
    l_colptr: &[usize],
    l_indices: &[usize],
    l_data: &[N],
    x: &mut [N])
where
N: Clone + Copy + Num {

    let n = l_indices.len();
    let lt = CsMat::from_slices(
        CompressedStorage::CSR, n, n, l_colptr, l_indices, l_data).unwrap();
    for (row_ind, vec) in lt.outer_iterator() {
        for (col_ind, value) in vec.iter() {
            x[row_ind] = x[row_ind] - value * x[col_ind];
        }
    }
}

pub fn ldl_dsolve<N>(
    d: &[N],
    x: &mut [N])
where
N: Clone + Copy + Num {

    for (xv, dv) in x.iter_mut().zip(d.iter()) {
        *xv = *xv / *dv;
    }
}

#[cfg(test)]
mod test {
    use sparse::csmat::CsMat;
    use sparse::csmat::CompressedStorage::{CSR};

    fn test_mat1() -> CsMat<f64, Vec<usize>, Vec<f64>> {
        let indptr = vec![0, 2, 5, 6, 7, 13, 14, 17, 20, 24, 28];
        let indices = vec![
            0, 8,
            1, 4, 9,
            2,
            3,
            1, 4, 6, 7, 8, 9,
            5,
            4, 6, 9,
            4, 7, 8,
            0, 4, 7, 8,
            1, 4, 6, 9];
        let data = vec![
            1.7, 0.13,
            1., 0.02, 0.01,
            1.5,
            1.1,
            0.02, 2.6, 0.16, 0.09, 0.52, 0.53,
            1.2,
            0.16, 1.3, 0.56,
            0.09, 1.6, 0.11,
            0.13, 0.52, 0.11, 1.4,
            0.01, 0.53, 0.56, 3.1];
        CsMat::from_vecs(CSR, 10, 10, indptr, indices, data).unwrap()
    }

    fn test_vec1() -> Vec<f64> {
        vec![0.287, 0.22, 0.45, 0.44, 2.486, 0.72 ,
             1.55 ,  1.424,1.621,  3.759]
    }

    fn expected_res1() -> Vec<f64> {
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    }
