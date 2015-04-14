/// Representation of permutation matrices
///
/// Both the permutation matrices and its inverse are stored

use std::ops::{Deref};

pub enum Permutation<IndStorage>
where IndStorage: Deref<Target=[usize]> {
    Identity,
    FinitePerm {
        perm: IndStorage,
        perm_inv: IndStorage,
    }
}

use self::Permutation::*;

impl Permutation<Vec<usize>> {

    pub fn new(perm: Vec<usize>) -> Permutation<Vec<usize>> {
        let mut perm_inv = perm.clone();
        for (ind, val) in perm.iter().enumerate() {
            perm_inv[*val] = ind;
        }
        FinitePerm {
            perm: perm,
            perm_inv: perm_inv
        }
    }
}

impl<IndStorage> Permutation<IndStorage>
where IndStorage: Deref<Target=[usize]> {

    pub fn identity() -> Permutation<IndStorage> {
        Identity
    }

    pub fn inv<'perm>(&'perm self) -> Permutation<&'perm [usize]> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p_[..], perm_inv: &p[..] }
        }
    }

    // TODO: either the trait Deref or Borrow should be implemnted for this
    pub fn borrowed<'perm>(&'perm self) -> Permutation<&'perm [usize]> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p[..], perm_inv: &p_[..] }
        }
    }

    pub fn owned_clone(&self) -> Permutation<Vec<usize>> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm {
                perm: p.iter().cloned().collect(),
                perm_inv: p_.iter().cloned().collect()
            }
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
