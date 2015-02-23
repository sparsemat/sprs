/// Cholesky factorization

use std::ops::{Deref};

use sparse::csmat::{CsMat};
use sparse::symmetric::{is_symmetric};
use sparse::permutation::Permutation;
use dense::vec;

use std::boxed::Box;

/// Optional workspace (to be moved in its own module)
pub enum OptWorkspace<T> {
    NoWorkspace,
    Workspace(T)
}

use self::OptWorkspace::*;

/// Result of the symbolic LDLt decomposition of a symmetric sparse matrix
struct SymbolicLDL {
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
fn ldl_symbolic<N, IStorage, DStorage, PStorage>(
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

    let mut parents = (0..n).map(|x| -1).collect::<Vec<isize>>();
    let mut l_nz = (0..n).map(|x| 0).collect::<Vec<usize>>();

    for (outer_ind, vec) in mat.outer_iterator_papt(&perm.borrowed()) {
        flag[outer_ind] = outer_ind; // this node is visited

        let perm_out = perm.at(outer_ind);

        for inner_ind in vec.indices().iter() {
            let mut perm_in = perm.at_inv(*inner_ind);

            if ( *inner_ind < outer_ind ) {
                // get back to the root of the etree
                // TODO: maybe this calls for a more adequate parent structure?
                while ( flag[perm_in] != outer_ind ) {
                    if ( parents[perm_in] == -1 ) {
                        parents[perm_in] = outer_ind as isize; // TODO check overflow
                    }
                    l_nz[perm_in] = l_nz[perm_in] + 1;
                    flag[perm_in] = outer_ind;
                    perm_in = parents[perm_in] as usize; // TODO check negative
                }
            }
        }
    }

    let mut l_colptr = Vec::<usize>::with_capacity(n+1);
    l_colptr.push(0);
    let mut prev : usize = 0;
    for k in (0..n) {
        prev = prev + l_nz[k];
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
