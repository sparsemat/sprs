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
pub type PermOwnedI<I> = Permutation<I, Vec<I>>;

pub type PermView<'a> = Permutation<usize, &'a [usize]>;
pub type PermViewI<'a, I> = Permutation<I, &'a [I]>;

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
    pub fn reborrow(&self) -> PermViewI<'a, I> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p[..], perm_inv: &p_[..] }
        }
    }

    pub fn reborrow_inv(&self) -> PermViewI<'a, I> {
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

    pub fn inv(&self) -> PermViewI<I> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p_[..], perm_inv: &p[..] }
        }
    }

    // TODO: either the trait Deref or Borrow should be implemnted for this
    pub fn view(&self) -> PermViewI<I> {
        match self {
            &Identity => Identity,
            &FinitePerm {
                perm: ref p, perm_inv: ref p_
            } => FinitePerm { perm: &p[..], perm_inv: &p_[..] }
        }
    }

    pub fn owned_clone(&self) -> PermOwnedI<I> {
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

    /// Get a vector representing this permutation
    ///
    /// n: dimension of the expected permutation matrix.
    pub fn vec(&self, n: I) -> Vec<I> {
        match self {
            &Identity => (0..n.index()).map(I::from_usize).collect(),
            &FinitePerm { perm: ref p, perm_inv: _} => {
                assert!(n.index() == p.len());
                p.to_vec()
            },
        }
    }

    /// Get a vector representing the inverse of this permutation
    pub fn inv_vec(&self, n: I) -> Vec<I> {
        match self {
            &Identity => (0..n.index()).map(I::from_usize).collect(),
            &FinitePerm { perm: _, perm_inv: ref p_} => {
                assert!(n.index() == p_.len());
                p_.to_vec()
            },
        }
    }

    pub fn to_other_idx_type<I2>(&self) -> PermOwnedI<I2>
    where I2: SpIndex
    {
        match self {
            &Identity => PermOwnedI::identity(),
            &FinitePerm { perm: ref p, perm_inv: ref p_ } => {
                let perm = p.iter()
                            .map(|i| I2::from_usize(i.index()))
                            .collect();
                let perm_inv = p_.iter()
                                 .map(|i| I2::from_usize(i.index()))
                                 .collect();
                FinitePerm {
                    perm: perm,
                    perm_inv: perm_inv,
                }
            },
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
