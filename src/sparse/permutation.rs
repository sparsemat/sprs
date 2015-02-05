/// Representation of permutation matrices
/// 
/// Both the permutation matrices and its inverse are stored

use std::borrow::{Cow, IntoCow};

#[derive(Clone)]
pub enum Permutation<'a> {
    Identity,
    FinitePerm {
        perm: Cow<'a, Vec<usize>, [usize]>,
        perm_inv: Cow<'a, Vec<usize>, [usize]>,
    }
}

use self::Permutation::*;

impl<'a> Permutation<'a> {

    pub fn new(perm: Vec<usize>) -> Permutation<'a> {
        let mut perm_inv = perm.clone().into_cow();
        for (ind, val) in perm.iter().enumerate() {
            perm_inv[*val] = ind;
        }
        FinitePerm {
            perm: perm.into_cow(),
            perm_inv: perm_inv
        }
    }

    pub fn identity() -> Permutation<'a> {
        Identity
    }

    pub fn inv(perm: Permutation<'a>) -> Permutation<'a> {
        match perm {
            Identity => Identity,
            FinitePerm {
                perm: p, perm_inv: p_ } => FinitePerm { perm: p_, perm_inv: p }
        }
    }

    pub fn inv_borrow(&self) -> Permutation<'a> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: p,
                perm_inv: p_ } => FinitePerm {
                          perm: p_.as_slice().into_cow(),
                          perm_inv: p.as_slice().into_cow() }
        }
    }

    pub fn at(&self, index: usize) -> usize {
        match self {
            &Identity => index,
            &FinitePerm {
                perm: p, perm_inv: _ } => p[index]
        }
    }

    pub fn at_inv(&self, index: usize) -> usize {
        match self {
            &Identity => index,
            &FinitePerm {
                perm: _, perm_inv: p_ } => p_[index]
        }
    }
}
