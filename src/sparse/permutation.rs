/// Representation of permutation matrices
///
/// Both the permutation matrices and its inverse are stored
use std::ops::{Deref, Mul};

use crate::dense_vector::{DenseVector, DenseVectorMut};
use crate::indexing::SpIndex;
use crate::sparse::{CompressedStorage, CsMatI, CsMatViewI};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

pub fn perm_is_valid<I: SpIndex>(perm: &[I]) -> bool {
    let n = perm.len();
    let mut seen = vec![false; n];
    for i in perm {
        if *i < I::zero() || *i >= I::from_usize(n) || seen[i.index()] {
            return false;
        }
        seen[i.index()] = true;
    }
    true
}

impl<I: SpIndex> PermOwnedI<I> {
    pub fn new(perm: Vec<I>) -> Self {
        assert!(perm_is_valid(&perm));
        Self::new_trusted(perm)
    }

    pub(crate) fn new_trusted(perm: Vec<I>) -> Self {
        let mut perm_inv = perm.clone();
        for (ind, val) in perm.iter().enumerate() {
            perm_inv[val.index()] = I::from_usize(ind);
        }
        Self {
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
                perm: p,
                perm_inv: p_,
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
                perm: p,
                perm_inv: p_,
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
    pub fn identity(dim: usize) -> Self {
        Self {
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

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Check whether the permutation is the identity.
    pub fn is_identity(&self) -> bool {
        match self.storage {
            Identity => true,
            FinitePerm {
                perm: ref p,
                perm_inv: ref _p_,
            } => p.iter().enumerate().all(|(ind, x)| ind == x.index()),
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

impl<'b, V, I, IndStorage> Mul<V> for &'b Permutation<I, IndStorage>
where
    IndStorage: 'b + Deref<Target = [I]>,
    V: DenseVector,
    <V as DenseVector>::Owned:
        DenseVectorMut + DenseVector<Scalar = <V as DenseVector>::Scalar>,
    <V as DenseVector>::Scalar: Clone,
    I: SpIndex,
{
    type Output = V::Owned;
    fn mul(self, rhs: V) -> Self::Output {
        assert_eq!(self.dim, rhs.dim());
        let mut res = rhs.to_owned();
        match self.storage {
            Identity => res,
            FinitePerm { perm: ref p, .. } => {
                for (i, pi) in p.iter().enumerate() {
                    *res.index_mut(i) = rhs.index(pi.index_unchecked()).clone();
                }
                res
            }
        }
    }
}

impl<V, I, IndStorage> Mul<V> for Permutation<I, IndStorage>
where
    IndStorage: Deref<Target = [I]>,
    V: DenseVector,
    <V as DenseVector>::Owned:
        DenseVectorMut + DenseVector<Scalar = <V as DenseVector>::Scalar>,
    <V as DenseVector>::Scalar: Clone,
    I: SpIndex,
{
    type Output = V::Owned;
    fn mul(self, rhs: V) -> Self::Output {
        &self * rhs
    }
}

impl<'a, N, I, Iptr, IndStorage> Mul<&'a Permutation<I, IndStorage>>
    for CsMatViewI<'a, N, I, Iptr>
where
    N: Clone + Default,
    I: SpIndex,
    Iptr: SpIndex,
    IndStorage: Deref<Target = [I]>,
{
    type Output = CsMatI<N, I, Iptr>;
    fn mul(self, perm: &'a Permutation<I, IndStorage>) -> CsMatI<N, I, Iptr> {
        assert!(self.cols() == perm.dim());
        if perm.is_identity() || self.rows() == 0 {
            return self.to_owned();
        }
        let mul_csc = |mat: CsMatViewI<_, _, _>| {
            let mut indptr = Vec::with_capacity(mat.indptr().len());
            let mut indices = Vec::with_capacity(mat.indices().len());
            let mut data = Vec::with_capacity(mat.data().len());
            let (_, pinv) = match perm.view().storage {
                Identity => unreachable!(),
                FinitePerm {
                    perm: p,
                    perm_inv: p_,
                } => (p, p_),
            };
            let mut nnz = Iptr::zero();
            indptr.push(nnz);
            for in_outer in pinv.iter() {
                nnz += mat.indptr().nnz_in_outer(in_outer.index());
                indptr.push(nnz);
                let outer = mat.outer_view(in_outer.index()).unwrap();
                indices.extend(outer.indices().iter().cloned());
                data.extend(outer.data().iter().cloned());
            }
            CsMatI::new_csc(mat.shape(), indptr, indices, data)
        };
        if self.is_csc() {
            mul_csc(self.view())
        } else {
            let res_csc = mul_csc(self.to_other_storage().view());
            // Question: should we respect the input storage, or always return
            // CSC?
            res_csc.to_other_storage()
        }
    }
}

/// Compute the square matrix resulting from the product P * A * P^T
pub fn transform_mat_papt<N, I, Iptr>(
    mat: CsMatViewI<N, I, Iptr>,
    perm: PermViewI<I>,
) -> CsMatI<N, I, Iptr>
where
    N: Clone + ::std::fmt::Debug,
    I: SpIndex,
    Iptr: SpIndex,
{
    assert!(mat.rows() == mat.cols());
    assert!(mat.rows() == perm.dim());
    if perm.is_identity() || mat.rows() == 0 {
        return mat.to_owned();
    }
    // We can apply the CSR algorithm even if A is CSC:
    // indeed, (PAP^T)^T = PA^TP^T, and transposing means going from CSC to CSR
    let mut indptr = Vec::with_capacity(mat.indptr().len());
    let mut indices = Vec::with_capacity(mat.indices().len());
    let mut data = Vec::with_capacity(mat.data().len());
    let (p, p_) = match perm.storage {
        Identity => unreachable!(),
        FinitePerm {
            perm: p,
            perm_inv: p_,
        } => (p, p_),
    };
    let mut nnz = Iptr::zero();
    indptr.push(nnz);
    let mut tmp = Vec::with_capacity(mat.max_outer_nnz());
    for in_outer in p {
        nnz += mat.indptr().nnz_in_outer(in_outer.index());
        indptr.push(nnz);
        tmp.clear();
        let outer = mat.outer_view(in_outer.index()).unwrap();
        for (ind, val) in outer.indices().iter().zip(outer.data()) {
            tmp.push((p_[ind.index()], val.clone()))
        }
        tmp.sort_by_key(|(ind, _)| *ind);
        for (ind, val) in &tmp {
            indices.push(*ind);
            data.push(val.clone());
        }
    }

    match mat.storage() {
        CompressedStorage::CSR => {
            CsMatI::new(mat.shape(), indptr, indices, data)
        }
        CompressedStorage::CSC => {
            CsMatI::new_csc(mat.shape(), indptr, indices, data)
        }
    }
}

/// Compute a permutation P that puts the requested indices at the
/// start of a matrix A of shape `(n, n)` when applying P * A * P^T
///
/// This can be useful to expose a block structure in a matrix.
///
/// # Failure
///
/// Returns `None` if an input index is not in the range `0..n`.
pub fn try_permute_to_top<I>(indices: &[I], n: usize) -> Option<PermOwnedI<I>>
where
    I: SpIndex,
{
    let mut used = vec![false; n];
    let mut perm = Vec::with_capacity(n);
    for i in indices {
        *used.get_mut(i.try_index()?)? = true;
        perm.push(*i);
    }
    perm.extend(
        used.iter()
            .enumerate()
            .filter(|(_, u)| !*u)
            .map(|(i, _)| I::from_usize(i)),
    );
    Some(PermOwnedI::new_trusted(perm))
}

/// Compute a permutation P that puts the requested indices at the
/// start of a matrix A of shape `(n, n)` when applying P * A * P^T
///
/// This can be useful to expose a block structure in a matrix.
///
/// # Panics
///
/// Panics if an input index is not in the range `0..n`.
pub fn permute_to_top<I>(indices: &[I], n: usize) -> PermOwnedI<I>
where
    I: SpIndex,
{
    try_permute_to_top(indices, n).expect("Out of bounds index")
}

#[cfg(test)]
mod test {
    use crate::sparse::CsMat;

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

        let x = ndarray::arr1(&[5, 1, 2, 3, 4]);
        let y = p.view() * x.view();
        assert_eq!(y, ndarray::arr1(&[2, 1, 3, 5, 4]));
    }

    #[test]
    fn mat_perm_mul() {
        // | 1 0 0 3 1 |
        // | 0 2 0 0 0 |
        // | 0 0 0 1 0 |
        // | 3 0 1 1 0 |
        // | 1 0 0 0 1 |
        let mat = CsMat::new_csc(
            (5, 5),
            vec![0, 3, 4, 5, 8, 10],
            vec![0, 3, 4, 1, 3, 0, 2, 3, 0, 4],
            vec![1, 3, 1, 2, 1, 3, 1, 1, 1, 1],
        );
        // | 0 0 1 0 0 |
        // | 0 1 0 0 0 |
        // | 0 0 0 1 0 |
        // | 1 0 0 0 0 |
        // | 0 0 0 0 1 |
        let perm = super::PermOwned::new(vec![2, 1, 3, 0, 4]);
        // expected matrix AP
        //                | 0 0 1 0 0 |
        //                | 0 1 0 0 0 |
        //                | 0 0 0 1 0 |
        //                | 1 0 0 0 0 |
        //                | 0 0 0 0 1 |
        //
        // | 1 0 0 3 1 |  | 3 0 1 0 1 |
        // | 0 2 0 0 0 |  | 0 2 0 0 0 |
        // | 0 0 0 1 0 |  | 1 0 0 0 0 |
        // | 3 0 1 1 0 |  | 1 0 3 1 0 |
        // | 1 0 0 0 1 |  | 0 0 1 0 1 |
        let expected = CsMat::new_csc(
            (5, 5),
            vec![0, 3, 4, 7, 8, 10],
            vec![0, 2, 3, 1, 0, 3, 4, 3, 0, 4],
            vec![3, 1, 1, 2, 1, 3, 1, 1, 1, 1],
        );
        let res = mat.view() * &perm;
        assert_eq!(res, expected);

        let mat = mat.to_csr();
        let expected = expected.to_csr();
        let res = mat.view() * &perm;
        assert_eq!(res, expected);
    }

    #[test]
    fn transform_mat_papt() {
        // | 1 0 0 3 1 |
        // | 0 2 0 0 0 |
        // | 0 0 0 1 0 |
        // | 3 0 1 1 0 |
        // | 1 0 0 0 1 |
        let mat = CsMat::new_csc(
            (5, 5),
            vec![0, 3, 4, 5, 8, 10],
            vec![0, 3, 4, 1, 3, 0, 2, 3, 0, 4],
            vec![1, 3, 1, 2, 1, 3, 1, 1, 1, 1],
        );

        let perm = super::PermOwned::new(vec![2, 1, 3, 0, 4]);
        // expected matrix PA
        // | 0 0 0 1 0 |
        // | 0 2 0 0 0 |
        // | 3 0 1 1 0 |
        // | 1 0 0 3 1 |
        // | 1 0 0 0 1 |
        // expected matrix PAP^T
        // | 0 0 1 0 0 |
        // | 0 2 0 0 0 |
        // | 1 0 1 3 0 |
        // | 0 0 3 1 1 |
        // | 0 0 0 1 1 |
        let expected_papt = CsMat::new_csc(
            (5, 5),
            vec![0, 1, 2, 5, 8, 10],
            vec![2, 1, 0, 2, 3, 2, 3, 4, 3, 4],
            vec![1, 2, 1, 1, 3, 3, 1, 1, 1, 1],
        );
        let papt = super::transform_mat_papt(mat.view(), perm.view());
        assert_eq!(expected_papt, papt);
    }

    #[test]
    fn permute_to_top() {
        let perm = super::permute_to_top(&[2, 3, 7, 8], 12);
        let expected =
            super::PermOwned::new(vec![2, 3, 7, 8, 0, 1, 4, 5, 6, 9, 10, 11]);
        assert_eq!(perm, expected);
    }

    #[test]
    fn perm_validity() {
        use super::perm_is_valid;
        assert!(perm_is_valid(&[0, 1, 2, 3, 4]));
        assert!(perm_is_valid(&[1, 0, 3, 4, 2]));
        assert!(!perm_is_valid(&[0, 1, 2, 3, 5]));
        assert!(!perm_is_valid(&[0, 1, 2, 3, 3]));
    }
}
