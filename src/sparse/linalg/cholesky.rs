/// Cholesky factorization

use std::ops::{Deref};

use num::traits::Num;

use sparse::csmat::{CsMat};
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
    perm: Permutation<PStorage>,
    flag: &mut [usize],
    check_symmetry: SymmetryCheck) -> Option<SymbolicLDL>
where
N: Clone + Copy + PartialEq,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]>,
PStorage: Deref<Target=[usize]> {

    match check_symmetry {
        SymmetryCheck::DontCheckSymmetry => (),
        SymmetryCheck::CheckSymmetry => if ! is_symmetric(mat) {
            return None;
        }
    }

    let n = mat.rows();

    let mut parents = vec![-1isize; n];
    let mut l_nz = vec![0usize; n];

    for (k, (outer_ind, vec)) in mat.outer_iterator_papt(&perm.borrowed()).enumerate() {

        flag[k] = k; // this node is visited

        for (inner_ind, _) in vec.iter() {
            let mut i = inner_ind;

            // FIXME: the article tests inner_ind versus k, but this looks
            // weird as it would introduce a dissimetry between the permuted
            // and non permuted cases. Needs test however
            if i < outer_ind {
                // get back to the root of the etree
                // TODO: maybe this calls for a more adequate parent structure?
                while flag[i] != outer_ind {
                    if parents[i] == -1 {
                        parents[i] = outer_ind as isize; // TODO check overflow
                    }
                    l_nz[i] = l_nz[i] + 1;
                    flag[i] = outer_ind;
                    i = parents[i] as usize; // TODO check negative
                }
            }
        }
    }

    let mut l_colptr = Vec::<usize>::with_capacity(n+1);
    l_colptr.push(0);
    let mut prev : usize = 0;
    for k in (0..n) {
        prev += l_nz[k];
        l_colptr.push(prev);
    }

    Some(SymbolicLDL {
        n: n,
        l_colptr: l_colptr,
        l_nz: l_nz,
        parents: parents,
        perm: perm.owned_clone()
    })
}

pub struct Ldlt<N> {
    l_nz: &[usize],
    l_indices: &[usize],
    l_data: &[N],
    d: &[N]
}

pub fn ldl_numeric<N, IStorage, DStorage>(
    mat: &CsMat<N, IStorage, DStorage>,
    ldl_sym: &SymbolicLDL,
    y_workspace: &mut [N],
    pattern_workspace: &mut [usize],
    flag_workspace: &mut [usize]) -> Result<(), usize>
where
N: Clone + Copy + PartialEq + Num,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]>,
PStorage: Deref<Target=[usize]> {
    panic!("not yet implemented");
    let n = ldl_sym.n;
    let parents = &ldl_sym.parents;
    let l_colptr = &ldl_sym.l_colptr;
    let mut l_nz = vec![0usize, n]; // FIXME: should leave allocation to caller
    let mut l_indices = vec![0usize, n]; // FIXME: should leave allocation to caller
    let mut diag = vec![N::zero(), n];

    for (k, (outer_ind, vec)) in mat.outer_iterator_papt(&perm.borrowed()).enumerate() {

        // compute the nonzero pattern of the kth row of L
        // in topological order

        flag[k] = k; // this node is visited
        y_workspace[k] = N::zero();
        l_nz[k] = 0;
        let mut top = n;

        for (inner_ind, val) in vec.iter().filter(|i| i <= k) {
            Y[inner_ind] += val;
            let mut i = inner_ind;
            let mut len = 0;
            while flag[i] != outer_ind {
                pattern_workspace[len] = i;
                len += 1;
                flag_workspace[i] = k;
                i = parents[i] as usize;
            }
            while len > 0 { // TODO: can be written as a loop with iterators
                top -= 1;
                len -= 1;
                pattern_workspace[top] = pattern_workspace[len];
            }
        }

        // use a sparse triangular solve to compute the values
        // of the kth row of L
        diag[k] = y_workspace[k];
        y_workspace[k] = N::zero();
        'pattern: for i in pattern[top..n] {
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
                Y[l_indices[p]] -= l_data[p] * yi;
            }
            let l_ki = yi / diag[i];
            diag[k] -= l_ki * yi;
            l_indices[p] = k;
            l_data[p] = l_ki;
            l_nz[i] += 1;
        }
        if diag[k] == N::zero() {
            return Err(k)
        }
    }
    Ok(())
}
