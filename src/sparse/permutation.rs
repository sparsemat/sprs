/// Representation of permutation matrices
/// 
/// Both the permutation matrices and its inverse are stored

#[derive(Clone)]
pub enum Permutation {
    Identity,
    FinitePerm {
        perm: Vec<usize>,
        perm_inv: Vec<usize>,
    }
}

use self::Permutation::*;

impl Permutation {

    pub fn new(perm: Vec<usize>) -> Permutation {
        let mut perm_inv = perm.clone();
        for (ind, val) in perm.iter().enumerate() {
            perm_inv[*val] = ind;
        }
        FinitePerm {
            perm: perm,
            perm_inv: perm_inv
        }
    }

    pub fn identity() -> Permutation {
        Identity
    }

    pub fn inv(perm: Permutation) -> Permutation {
        match perm {
            Identity => Identity,
            FinitePerm {
                perm: p, perm_inv: p_ } => FinitePerm { perm: p_, perm_inv: p }
        }
    }

    pub fn at(&self, index: usize) -> usize {
        match self {
            &Identity => index,
            &FinitePerm {
                perm: ref p, perm_inv: _ } => p[index]
        }
    }

    pub fn at_inv(&self, index: usize) -> usize {
        match self {
            &Identity => index,
            &FinitePerm {
                perm: _, perm_inv: ref p_ } => p_[index]
        }
    }
}
