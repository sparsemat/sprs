/// Representation of permutation matrices
///
/// Both the permutation matrices and its inverse are stored

use std::ops::{Deref, Mul};

#[derive(Debug)]
pub enum Permutation<IndStorage>
where IndStorage: Deref<Target=[usize]> {
    Identity,
    FinitePerm {
        perm: IndStorage,
        perm_inv: IndStorage,
    }
}

pub type PermOwned = Permutation<Vec<usize>>;
pub type PermView<'a> = Permutation<&'a [usize]>;

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

impl<'a> Permutation<&'a [usize]> {
    pub fn reborrow(&self) -> PermView<'a> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p[..], perm_inv: &p_[..] }
        }
    }

    pub fn reborrow_inv(&self) -> PermView<'a> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p_[..], perm_inv: &p[..] }
        }
    }
}

impl<IndStorage> Permutation<IndStorage>
where IndStorage: Deref<Target=[usize]> {

    pub fn identity() -> Permutation<IndStorage> {
        Identity
    }

    pub fn inv(&self) -> PermView {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p_[..], perm_inv: &p[..] }
        }
    }

    // TODO: either the trait Deref or Borrow should be implemnted for this
    pub fn view(&self) -> PermView {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p[..], perm_inv: &p_[..] }
        }
    }

    pub fn owned_clone(&self) -> PermOwned {
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

impl<'a, 'b, N, IndStorage> Mul<&'a [N]> for &'b Permutation<IndStorage>
where IndStorage: 'b + Deref<Target=[usize]>,
      N: 'a + Copy
{
    type Output = Vec<N>;
    fn mul(self, rhs: &'a [N]) -> Vec<N> {
        let mut res = rhs.to_vec();
        match self {
            &Identity => res,
            &FinitePerm {
                perm: ref p,
                perm_inv: _,
            } => {
                for (&pi, r) in p.iter().zip(res.iter_mut()) {
                    *r = rhs[pi];
                }
                res
            }
        }
    }
}


mod test {

    #[test]
    fn perm_mul() {
        // |0 0 1 0 0| |5|   |2|
        // |0 1 0 0 0| |1|   |1|
        // |0 0 0 1 0| |2| = |3|
        // |1 0 0 0 0| |3|   |5|
        // |0 0 0 0 1| |4|   |4|
        let x = vec![5, 1, 2, 3, 4];
        let p = super::PermOwned::new(vec![2, 1, 3, 0, 4]);

        let y = &p * &x;
        assert_eq!(&y, &[2, 1, 3, 5, 4]);
    }
}
