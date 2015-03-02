/// Cholesky factorization

use std::ops::{Deref};

use sparse::csmat::{CsMat};
use sparse::symmetric::{is_symmetric};
use sparse::permutation::Permutation;

use std::boxed::Box;

/// Optional workspace (to be moved in its own module)
pub enum OptWorkspace<T> {
    NoWorkspace,
    Workspace(T)
}

use self::OptWorkspace::*;

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

/// Perform a symbolic LDLt decomposition of a symmetric sparse matrix
pub fn ldl_symbolic<N, IStorage, DStorage, PStorage>(
    mat: &CsMat<N, IStorage, DStorage>,
    perm: Permutation<PStorage>,
    flag_workspace: OptWorkspace<&mut [usize]>) -> Option<SymbolicLDL>
where
N: Clone + Copy + PartialEq,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]>,
PStorage: Deref<Target=[usize]> {
    if ! is_symmetric(mat) {
        return None;
    }

    let n = mat.rows();

    let mut ws = Box::new(Vec::<usize>::new());
    let mut flag = match flag_workspace {
        NoWorkspace => {
            for _ in (0..n) {
                ws.push(0);
            }
            ws.as_mut_slice()
        },
        Workspace(w) => w
    };

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

pub struct LDLT {
    tmp: usize // TODO
}

pub fn ldl_numeric<N, IStorage, DStorage, PStorage>(
    mat: &CsMat<N, IStorage, DStorage>,
    ldl_sym: SymbolicLDL,
    perm: Permutation<PStorage>,
    y_workspace: OptWorkspace<&mut[N]>,
    pattern_workspace: OptWorkspace<&mut[usize]>,
    flag_workspace: OptWorkspace<&mut[usize]>) -> LDLT
where
N: Clone + Copy + PartialEq,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]>,
PStorage: Deref<Target=[usize]> {
    panic!("not yet implemented");
    LDLT {
        tmp: 1
    }
}
