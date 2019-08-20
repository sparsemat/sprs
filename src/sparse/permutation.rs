use indexing::SpIndex;
/// Representation of permutation matrices
///
/// Both the permutation matrices and its inverse are stored
use std::ops::{Deref, Mul};

#[derive(Debug, Clone)]
enum PermStorage<I, IndStorage>
where
    IndStorage: Deref<Target = [I]>,
{
    Identity,
    FinitePerm {
        perm: IndStorage,
        perm_inv: IndStorage,
    },
}

use self::PermStorage::{FinitePerm, Identity};

#[derive(Debug, Clone)]
pub struct Permutation<I, IndStorage>
where
    IndStorage: Deref<Target = [I]>,
{
    dim: usize,
    storage: PermStorage<I, IndStorage>,
}

pub type PermOwned = Permutation<usize, Vec<usize>>;
pub type PermOwnedI<I> = Permutation<I, Vec<I>>;

pub type PermView<'a> = Permutation<usize, &'a [usize]>;
pub type PermViewI<'a, I> = Permutation<I, &'a [I]>;

impl<I: SpIndex> Permutation<I, Vec<I>> {
    pub fn new(perm: Vec<I>) -> PermOwnedI<I> {
        let mut perm_inv = perm.clone();
        for (ind, val) in perm.iter().enumerate() {
            perm_inv[val.index()] = I::from_usize(ind);
        }
        PermOwnedI {
            dim: perm.len(),
            storage: FinitePerm { perm, perm_inv },
        }
    }
}

impl<'a, I: SpIndex> Permutation<I, &'a [I]> {
    pub fn reborrow(&self) -> PermViewI<'a, I> {
        match self.storage {
            Identity => PermViewI {
                dim: self.dim,
                storage: Identity,
            },
            FinitePerm {
                perm: ref p,
                perm_inv: ref p_,
            } => PermViewI {
                dim: self.dim,
                storage: FinitePerm {
                    perm: &p[..],
                    perm_inv: &p_[..],
                },
            },
        }
    }

    pub fn reborrow_inv(&self) -> PermViewI<'a, I> {
        match self.storage {
            Identity => PermViewI {
                dim: self.dim,
                storage: Identity,
            },
            FinitePerm {
                perm: ref p,
                perm_inv: ref p_,
            } => PermViewI {
                dim: self.dim,
                storage: FinitePerm {
                    perm: &p_[..],
                    perm_inv: &p[..],
                },
            },
        }
    }
}

impl<I: SpIndex, IndStorage> Permutation<I, IndStorage>
where
    IndStorage: Deref<Target = [I]>,
{
    pub fn identity(dim: usize) -> Permutation<I, IndStorage> {
        Permutation {
            dim,
            storage: Identity,
        }
    }

    pub fn inv(&self) -> PermViewI<I> {
        match self.storage {
            Identity => PermViewI {
                dim: self.dim,
                storage: Identity,
            },
            FinitePerm {
                perm: ref p,
                perm_inv: ref p_,
            } => PermViewI {
                dim: self.dim,
                storage: FinitePerm {
                    perm: &p_[..],
                    perm_inv: &p[..],
                },
            },
        }
    }

    pub fn view(&self) -> PermViewI<I> {
        match self.storage {
            Identity => PermViewI {
                dim: self.dim,
                storage: Identity,
            },
            FinitePerm {
                perm: ref p,
                perm_inv: ref p_,
            } => PermViewI {
                dim: self.dim,
                storage: FinitePerm {
                    perm: &p[..],
                    perm_inv: &p_[..],
                },
            },
        }
    }

    pub fn owned_clone(&self) -> PermOwnedI<I> {
        match self.storage {
            Identity => PermOwnedI {
                dim: self.dim,
                storage: Identity,
            },
            FinitePerm {
                perm: ref p,
                perm_inv: ref p_,
            } => PermOwnedI {
                dim: self.dim,
                storage: FinitePerm {
                    perm: p.iter().cloned().collect(),
                    perm_inv: p_.iter().cloned().collect(),
                },
            },
        }
    }

    pub fn at(&self, index: usize) -> usize {
        assert!(index < self.dim);
        match self.storage {
            Identity => index,
            FinitePerm { perm: ref p, .. } => p[index].index_unchecked(),
        }
    }

    pub fn at_inv(&self, index: usize) -> usize {
        assert!(index < self.dim);
        match self.storage {
            Identity => index,
            FinitePerm {
                perm_inv: ref p_, ..
            } => p_[index].index_unchecked(),
        }
    }

    /// Get a vector representing this permutation
    pub fn vec(&self) -> Vec<I> {
        match self.storage {
            Identity => (0..self.dim).map(I::from_usize).collect(),
            FinitePerm { perm: ref p, .. } => p.to_vec(),
        }
    }

    /// Get a vector representing the inverse of this permutation
    pub fn inv_vec(&self) -> Vec<I> {
        match self.storage {
            Identity => (0..self.dim).map(I::from_usize).collect(),
            FinitePerm {
                perm_inv: ref p_, ..
            } => p_.to_vec(),
        }
    }

    pub fn to_other_idx_type<I2>(&self) -> PermOwnedI<I2>
    where
        I2: SpIndex,
    {
        match self.storage {
            Identity => PermOwnedI::identity(self.dim),
            FinitePerm {
                perm: ref p,
                perm_inv: ref p_,
            } => {
                let perm = p
                    .iter()
                    .map(|i| I2::from_usize(i.index_unchecked()))
                    .collect();
                let perm_inv = p_
                    .iter()
                    .map(|i| I2::from_usize(i.index_unchecked()))
                    .collect();
                PermOwnedI {
                    dim: self.dim,
                    storage: FinitePerm { perm, perm_inv },
                }
            }
        }
    }
}

impl<'a, 'b, N, I, IndStorage> Mul<&'a [N]> for &'b Permutation<I, IndStorage>
where
    IndStorage: 'b + Deref<Target = [I]>,
    N: 'a + Copy,
    I: SpIndex,
{
    type Output = Vec<N>;
    fn mul(self, rhs: &'a [N]) -> Vec<N> {
        assert_eq!(self.dim, rhs.len());
        let mut res = rhs.to_vec();
        match self.storage {
            Identity => res,
            FinitePerm { perm: ref p, .. } => {
                for (pi, r) in p.iter().zip(res.iter_mut()) {
                    *r = rhs[pi.index_unchecked()];
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
