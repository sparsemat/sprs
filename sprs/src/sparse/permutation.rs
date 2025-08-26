/// Representation of permutation matrices
///
/// Both the permutation matrices and its inverse are stored
use std::ops::{Deref, Mul};

use crate::dense_vector::{DenseVector, DenseVectorMut};
use crate::indexing::SpIndex;
use crate::sparse::{CompressedStorage, CsMatI, CsMatViewI};

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

impl<I: SpIndex> PermViewI<'_, I> {
    pub fn reborrow(&self) -> Self {
        match self.storage {
            Identity => Self {
                dim: self.dim,
                storage: Identity,
            },
            FinitePerm {
                perm: p,
                perm_inv: p_,
            } => Self {
                dim: self.dim,
                storage: FinitePerm {
                    perm: &p[..],
                    perm_inv: &p_[..],
                },
            },
        }
    }

    pub fn reborrow_inv(&self) -> Self {
        match self.storage {
            Identity => Self {
                dim: self.dim,
                storage: Identity,
            },
            FinitePerm {
                perm: p,
                perm_inv: p_,
            } => Self {
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

    pub fn inv(&self) -> PermViewI<'_, I> {
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

    pub fn view(&self) -> PermViewI<'_, I> {
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
                    perm: p.iter().copied().collect(),
                    perm_inv: p_.iter().copied().collect(),
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

/// Compute the outer (uncompressed) dimension only
fn permute_outer<N, I, Iptr>(
    mat: CsMatViewI<N, I, Iptr>,
    perm: PermViewI<I>,
) -> CsMatI<N, I, Iptr>
where
    N: Clone + ::std::fmt::Debug,
    I: SpIndex,
    Iptr: SpIndex,
{
    assert!(mat.outer_dims() == perm.dim());
    if mat.rows() == 0 || mat.cols() == 0 {
        return mat.to_owned();
    }

    let mut indptr = Vec::with_capacity(mat.indptr().len());
    let mut indices = Vec::with_capacity(mat.indices().len());
    let mut data = Vec::with_capacity(mat.data().len());

    let p = match perm.storage {
        Identity => unreachable!(),
        FinitePerm {
            perm: p,
            perm_inv: _,
        } => p,
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
            tmp.push((*ind, val.clone()));
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

/// Compute the inner (compressed) dimension only
fn permute_inner<N, I, Iptr>(
    mat: CsMatViewI<N, I, Iptr>,
    perm: PermViewI<I>,
) -> CsMatI<N, I, Iptr>
where
    N: Clone + ::std::fmt::Debug,
    I: SpIndex,
    Iptr: SpIndex,
{
    assert!(mat.inner_dims() == perm.dim());
    if mat.rows() == 0 || mat.cols() == 0 {
        return mat.to_owned();
    }

    let mut indptr = Vec::with_capacity(mat.indptr().len());
    let mut indices = Vec::with_capacity(mat.indices().len());
    let mut data = Vec::with_capacity(mat.data().len());
    let p_ = match perm.storage {
        Identity => unreachable!(),
        FinitePerm {
            perm: _,
            perm_inv: p_,
        } => p_,
    };

    let mut nnz = Iptr::zero();
    indptr.push(nnz);
    let mut tmp = Vec::with_capacity(mat.max_outer_nnz());
    for in_outer in 0..mat.outer_dims() {
        nnz += mat.indptr().nnz_in_outer(in_outer.index());
        indptr.push(nnz);
        tmp.clear();

        let outer = mat.outer_view(in_outer.index()).unwrap();
        for (ind, val) in outer.indices().iter().zip(outer.data()) {
            tmp.push((p_[ind.index()], val.clone()));
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

/// Compute the matrix resulting from the product P * A
pub fn permute_rows<N, I, Iptr>(
    mat: CsMatViewI<N, I, Iptr>,
    perm: PermViewI<I>,
) -> CsMatI<N, I, Iptr>
where
    N: Clone + ::std::fmt::Debug,
    I: SpIndex,
    Iptr: SpIndex,
{
    match mat.storage {
        CompressedStorage::CSC => permute_inner(mat, perm),
        CompressedStorage::CSR => permute_outer(mat, perm),
    }
}

/// Compute the matrix resulting from the product A * P
pub fn permute_cols<N, I, Iptr>(
    mat: CsMatViewI<N, I, Iptr>,
    perm: PermViewI<I>,
) -> CsMatI<N, I, Iptr>
where
    N: Clone + ::std::fmt::Debug,
    I: SpIndex,
    Iptr: SpIndex,
{
    match mat.storage {
        CompressedStorage::CSC => permute_outer(mat, perm),
        CompressedStorage::CSR => permute_inner(mat, perm),
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
            tmp.push((p_[ind.index()], val.clone()));
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

/// Compute the matrix resulting from the product P * A * Q,
/// using the same number of operations and allocation as
/// for computing either P * A or A * Q by itself.
pub fn transform_mat_paq<N, I, Iptr>(
    mat: CsMatViewI<N, I, Iptr>,
    row_perm: PermViewI<I>,
    col_perm: PermViewI<I>,
) -> CsMatI<N, I, Iptr>
where
    N: Clone + ::std::fmt::Debug,
    I: SpIndex,
    Iptr: SpIndex,
{
    assert!(mat.rows() == row_perm.dim());
    assert!(mat.cols() == col_perm.dim());

    if (row_perm.is_identity() && col_perm.is_identity())
        || mat.rows() == 0
        || mat.cols() == 0
    {
        return mat.to_owned();
    }

    // Unpack permutations and handle the case where one or the other is identity.
    // Similar to the PAP^T variant, we can use the same method for CSC and CSR
    // but unlike the PAP^T variant, we have to account for identity permutations
    // since one of P or Q may be identity when the other is not, and we have to swap
    // the permutations depending on which storage medium we're using
    let (p, p_) = match row_perm.storage {
        Identity => {
            // If the row permutation is identity, only do a column permutation
            return permute_cols(mat, col_perm);
        }
        FinitePerm {
            perm: p,
            perm_inv: p_,
        } => (p, p_),
    };

    let (q, q_) = match col_perm.storage {
        Identity => {
            // If the column permutation is identity, only do a row permutation
            return permute_rows(mat, row_perm);
        }
        FinitePerm {
            perm: q,
            perm_inv: q_,
        } => (q, q_),
    };

    // Switch permutations between inner and outer depending on storage medium
    let (outer_perm, inner_perm) = match mat.storage() {
        CompressedStorage::CSR => (p, q_),
        CompressedStorage::CSC => (q, p_),
    };

    // If both permutations are non-identity, apply them together in about the
    // same number of operations as applying just one or the other, following
    // the same pattern as transform_mat_papt().
    let mut indptr = Vec::with_capacity(mat.indptr().len());
    let mut indices = Vec::with_capacity(mat.indices().len());
    let mut data = Vec::with_capacity(mat.data().len());
    let mut nnz = Iptr::zero();
    indptr.push(nnz);
    let mut tmp = Vec::with_capacity(mat.max_outer_nnz());
    for in_outer in outer_perm {
        nnz += mat.indptr().nnz_in_outer(in_outer.index());
        indptr.push(nnz);
        tmp.clear();
        let outer = mat.outer_view(in_outer.index()).unwrap();
        for (ind, val) in outer.indices().iter().zip(outer.data()) {
            tmp.push((inner_perm[ind.index()], val.clone()));
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
    fn transform_mat_paq() {
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

        let row_perm = super::PermOwned::new(vec![2, 1, 3, 0, 4]);
        let col_perm = super::PermOwned::new(vec![1, 2, 3, 0, 4]);
        // expected matrix PAQ
        // | 0 0 1 0 0 |
        // | 2 0 0 0 0 |
        // | 0 1 1 3 0 |
        // | 0 0 3 1 1 |
        // | 0 0 0 1 1 |
        let expected_paq = CsMat::new_csc(
            (5, 5),
            vec![0, 1, 2, 5, 8, 10],
            vec![1, 2, 0, 2, 3, 2, 3, 4, 3, 4],
            vec![2, 1, 1, 1, 3, 3, 1, 1, 1, 1],
        );

        // Test with CSC
        let paq = super::transform_mat_paq(
            mat.view(),
            row_perm.view(),
            col_perm.view(),
        );
        assert_eq!(expected_paq.to_dense(), paq.to_dense());

        // Test with CSR
        let paq = super::transform_mat_paq(
            mat.to_other_storage().view(),
            row_perm.view(),
            col_perm.view(),
        );
        assert_eq!(expected_paq.to_dense(), paq.to_dense());

        // Make sure we get the same result as applying the permutations separately
        let aq = super::permute_cols(mat.view(), col_perm.view());
        let paq_separate = super::permute_rows(aq.view(), row_perm.view());
        assert_eq!(expected_paq.to_dense(), paq_separate.to_dense());
    }

    #[test]
    fn permute_rows() {
        // Test with nonsquare matrix to make sure dimensions are handled properly

        // | 1 0 0 3 |
        // | 0 2 0 0 |
        // | 0 0 0 1 |
        // | 3 0 1 1 |
        // | 1 0 0 0 |
        let mat = CsMat::new_csc(
            (5, 4),
            vec![0, 3, 4, 5, 8],
            vec![0, 3, 4, 1, 3, 0, 2, 3],
            vec![1, 3, 1, 2, 1, 3, 1, 1],
        );

        // | 0 2 0 0 |
        // | 1 0 0 3 |
        // | 0 0 0 1 |
        // | 3 0 1 1 |
        // | 1 0 0 0 |
        let pa_expected = CsMat::new_csc(
            (5, 4),
            vec![0, 3, 4, 5, 8],
            vec![1, 3, 4, 0, 3, 1, 2, 3],
            vec![1, 3, 1, 2, 1, 3, 1, 1],
        );

        // Test with CSC
        let perm = super::PermOwned::new(vec![1, 0, 2, 3, 4]);
        let pa = super::permute_rows(mat.view(), perm.view());
        assert_eq!(pa_expected.to_dense(), pa.to_dense());

        // Test with CSR
        let perm = super::PermOwned::new(vec![1, 0, 2, 3, 4]);
        let pa =
            super::permute_rows(mat.to_other_storage().view(), perm.view());
        assert_eq!(pa_expected.to_dense(), pa.to_dense());
    }

    #[test]
    fn permute_cols() {
        // Test with nonsquare matrix to make sure dimensions are handled properly

        // | 1 0 0 3 |
        // | 0 2 0 0 |
        // | 0 0 0 1 |
        // | 3 0 1 1 |
        // | 1 0 0 0 |
        let mat = CsMat::new_csc(
            (5, 4),
            vec![0, 3, 4, 5, 8],
            vec![0, 3, 4, 1, 3, 0, 2, 3],
            vec![1, 3, 1, 2, 1, 3, 1, 1],
        );

        // | 0 2 0 0 |
        // | 1 0 0 3 |
        // | 0 0 0 1 |
        // | 3 0 1 1 |
        // | 1 0 0 0 |
        let atqt_expected = CsMat::new_csc(
            (5, 4),
            vec![0, 3, 4, 5, 8],
            vec![1, 3, 4, 0, 3, 1, 2, 3],
            vec![1, 3, 1, 2, 1, 3, 1, 1],
        );

        // Test with CSC
        let perm = super::PermOwned::new(vec![1, 0, 2, 3, 4]);
        let atq = super::permute_cols(mat.transpose_view(), perm.view());
        assert_eq!(atqt_expected.to_dense(), atq.transpose_view().to_dense());

        // Test with CSR
        let perm = super::PermOwned::new(vec![1, 0, 2, 3, 4]);
        let atq = super::permute_cols(
            mat.to_other_storage().transpose_view(),
            perm.view(),
        );
        assert_eq!(atqt_expected.to_dense(), atq.transpose_view().to_dense());
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
