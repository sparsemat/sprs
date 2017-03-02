/// Representation of permutation matrices
///
/// Both the permutation matrices and its inverse are stored

use std::ops::{Deref, Mul};
use indexing::SpIndex;

#[derive(Debug, Clone)]
pub enum Permutation<I, IndStorage>
where IndStorage: Deref<Target=[I]> {
    Identity,
    FinitePerm {
        perm: IndStorage,
        perm_inv: IndStorage,
    }
}

pub type PermOwned = Permutation<usize, Vec<usize>>;
pub type PermOwned_<I> = Permutation<I, Vec<I>>;

pub type PermView<'a> = Permutation<usize, &'a [usize]>;
pub type PermView_<'a, I> = Permutation<I, &'a [I]>;

use self::Permutation::*;

impl<I: SpIndex> Permutation<I, Vec<I>> {

    pub fn new(perm: Vec<I>) -> Permutation<I, Vec<I>> {
        let mut perm_inv = perm.clone();
        for (ind, val) in perm.iter().enumerate() {
            perm_inv[val.index()] = I::from_usize(ind);
        }
        FinitePerm {
            perm: perm,
            perm_inv: perm_inv
        }
    }
}

impl<'a, I: SpIndex> Permutation<I, &'a [I]> {
    pub fn reborrow(&self) -> PermView_<'a, I> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p[..], perm_inv: &p_[..] }
        }
    }

    pub fn reborrow_inv(&self) -> PermView_<'a, I> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p_[..], perm_inv: &p[..] }
        }
    }
}

impl<I: SpIndex, IndStorage> Permutation<I, IndStorage>
where IndStorage: Deref<Target=[I]> {

    pub fn identity() -> Permutation<I, IndStorage> {
        Identity
    }

    pub fn inv(&self) -> PermView_<I> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p_[..], perm_inv: &p[..] }
        }
    }

    // TODO: either the trait Deref or Borrow should be implemnted for this
    pub fn view(&self) -> PermView_<I> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p[..], perm_inv: &p_[..] }
        }
    }

    pub fn owned_clone(&self) -> PermOwned_<I> {
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
                perm: ref p, perm_inv: _ } => p[index].index()
        }
    }

    pub fn at_inv(&self, index: usize) -> usize {
        match self {
            &Identity => index,
            &FinitePerm {
                perm: _, perm_inv: ref p_ } => p_[index].index()
        }
    }
}

impl<'a, 'b, N, I, IndStorage> Mul<&'a [N]> for &'b Permutation<I, IndStorage>
where IndStorage: 'b + Deref<Target=[I]>,
      N: 'a + Copy,
      I: SpIndex
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
                for (pi, r) in p.iter().zip(res.iter_mut()) {
                    *r = rhs[pi.index()];
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
