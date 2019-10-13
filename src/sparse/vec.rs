use ndarray::{self, ArrayBase};
use std::cmp;
use std::collections::HashSet;
use std::convert::AsRef;
use std::hash::Hash;
/// A sparse vector, which can be extracted from a sparse matrix
///
/// # Example
/// ```rust
/// use sprs::CsVec;
/// let vec1 = CsVec::new(8, vec![0, 2, 5, 6], vec![1.; 4]);
/// let vec2 = CsVec::new(8, vec![1, 3, 5], vec![2.; 3]);
/// let res = &vec1 + &vec2;
/// let mut iter = res.iter();
/// assert_eq!(iter.next(), Some((0, &1.)));
/// assert_eq!(iter.next(), Some((1, &2.)));
/// assert_eq!(iter.next(), Some((2, &1.)));
/// assert_eq!(iter.next(), Some((3, &2.)));
/// assert_eq!(iter.next(), Some((5, &3.)));
/// assert_eq!(iter.next(), Some((6, &1.)));
/// assert_eq!(iter.next(), None);
/// ```
use std::iter::{Enumerate, FilterMap, IntoIterator, Peekable, Sum, Zip};
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Neg, Sub};
use std::slice::{self, Iter, IterMut};
use Ix1;

use num_traits::{Float, Num, Signed, Zero};

use array_backend::Array2;
use errors::SprsError;
use indexing::SpIndex;
use sparse::csmat::CompressedStorage::{CSC, CSR};
use sparse::permutation::PermViewI;
use sparse::prelude::*;
use sparse::utils;
use sparse::{binop, prod};

impl<IS: Copy, DS: Copy> Copy for CsVecBase<IS, DS> {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// Hold the index of a non-zero element in the compressed storage
///
/// An NnzIndex can be used to later access the non-zero element in constant
/// time.
pub struct NnzIndex(pub usize);

/// A trait to represent types which can be interpreted as vectors
/// of a given dimension.
pub trait VecDim<N> {
    /// The dimension of the vector
    fn dim(&self) -> usize;
}

impl<N, IS, DS: Deref<Target = [N]>> VecDim<N> for CsVecBase<IS, DS> {
    fn dim(&self) -> usize {
        self.dim
    }
}

impl<N, T: ?Sized> VecDim<N> for T
where
    T: AsRef<[N]>,
{
    fn dim(&self) -> usize {
        self.as_ref().len()
    }
}

/// An iterator over the non-zero elements of a sparse vector
pub struct VectorIterator<'a, N: 'a, I: 'a> {
    ind_data: Zip<Iter<'a, I>, Iter<'a, N>>,
}

pub struct VectorIteratorPerm<'a, N: 'a, I: 'a> {
    ind_data: Zip<Iter<'a, I>, Iter<'a, N>>,
    perm: PermViewI<'a, I>,
}

/// An iterator over the mutable non-zero elements of a sparse vector
pub struct VectorIteratorMut<'a, N: 'a, I: 'a> {
    ind_data: Zip<Iter<'a, I>, IterMut<'a, N>>,
}

impl<'a, N: 'a, I: 'a + SpIndex> Iterator for VectorIterator<'a, N, I> {
    type Item = (usize, &'a N);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.ind_data.next() {
            None => None,
            Some((inner_ind, data)) => {
                Some((inner_ind.index_unchecked(), data))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ind_data.size_hint()
    }
}

impl<'a, N: 'a, I: 'a + SpIndex> Iterator for VectorIteratorPerm<'a, N, I> {
    type Item = (usize, &'a N);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.ind_data.next() {
            None => None,
            Some((inner_ind, data)) => {
                Some((self.perm.at(inner_ind.index_unchecked()), data))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ind_data.size_hint()
    }
}

impl<'a, N: 'a, I: 'a + SpIndex> Iterator for VectorIteratorMut<'a, N, I> {
    type Item = (usize, &'a mut N);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.ind_data.next() {
            None => None,
            Some((inner_ind, data)) => {
                Some((inner_ind.index_unchecked(), data))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ind_data.size_hint()
    }
}

pub trait SparseIterTools: Iterator {
    /// Iterate over non-zero elements of either of two vectors.
    /// This is useful for implementing eg addition of vectors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sprs::CsVec;
    /// use sprs::vec::NnzEither;
    /// use sprs::vec::SparseIterTools;
    /// let v0 = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
    /// let v1 = CsVec::new(5, vec![1, 2, 3], vec![-1., -2., -3.]);
    /// let mut nnz_or_iter = v0.iter().nnz_or_zip(v1.iter());
    /// assert_eq!(nnz_or_iter.next(), Some(NnzEither::Left((0, &1.))));
    /// assert_eq!(nnz_or_iter.next(), Some(NnzEither::Right((1, &-1.))));
    /// assert_eq!(nnz_or_iter.next(), Some(NnzEither::Both((2, &2., &-2.))));
    /// assert_eq!(nnz_or_iter.next(), Some(NnzEither::Right((3, &-3.))));
    /// assert_eq!(nnz_or_iter.next(), Some(NnzEither::Left((4, &3.))));
    /// assert_eq!(nnz_or_iter.next(), None);
    /// ```
    fn nnz_or_zip<'a, I, N1, N2>(
        self,
        other: I,
    ) -> NnzOrZip<'a, Self, I::IntoIter, N1, N2>
    where
        Self: Iterator<Item = (usize, &'a N1)> + Sized,
        I: IntoIterator<Item = (usize, &'a N2)>,
    {
        NnzOrZip {
            left: self.peekable(),
            right: other.into_iter().peekable(),
            life: PhantomData,
        }
    }

    /// Iterate over the matching non-zero elements of both vectors
    /// Useful for vector dot product.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sprs::CsVec;
    /// use sprs::vec::SparseIterTools;
    /// let v0 = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
    /// let v1 = CsVec::new(5, vec![1, 2, 3], vec![-1., -2., -3.]);
    /// let mut nnz_zip = v0.iter().nnz_zip(v1.iter());
    /// assert_eq!(nnz_zip.next(), Some((2, &2., &-2.)));
    /// assert_eq!(nnz_zip.next(), None);
    /// ```
    fn nnz_zip<'a, I, N1, N2>(
        self,
        other: I,
    ) -> FilterMap<
        NnzOrZip<'a, Self, I::IntoIter, N1, N2>,
        fn(NnzEither<'a, N1, N2>) -> Option<(usize, &'a N1, &'a N2)>,
    >
    where
        Self: Iterator<Item = (usize, &'a N1)> + Sized,
        I: IntoIterator<Item = (usize, &'a N2)>,
    {
        let nnz_or_iter = NnzOrZip {
            left: self.peekable(),
            right: other.into_iter().peekable(),
            life: PhantomData,
        };
        nnz_or_iter.filter_map(filter_both_nnz)
    }
}

impl<T: Iterator> SparseIterTools for Enumerate<T> {}

impl<'a, N: 'a, I: 'a + SpIndex> SparseIterTools for VectorIterator<'a, N, I> {}

/// Trait for types that can be iterated as sparse vectors
pub trait IntoSparseVecIter<'a, N: 'a> {
    type IterType;

    /// Transform self into an iterator that yields (usize, &N) tuples
    /// where the usize is the index of the value in the sparse vector.
    /// The indices should be sorted.
    fn into_sparse_vec_iter(
        self,
    ) -> <Self as IntoSparseVecIter<'a, N>>::IterType
    where
        <Self as IntoSparseVecIter<'a, N>>::IterType:
            Iterator<Item = (usize, &'a N)>;

    /// The dimension of the vector
    fn dim(&self) -> usize;

    /// Indicator to check whether the vector is actually dense
    fn is_dense(&self) -> bool {
        false
    }

    /// Random access to an element in the vector.
    ///
    /// # Panics
    ///
    /// - if the vector is not dense
    /// - if the index is out of bounds
    #[allow(unused_variables)]
    fn index(self, idx: usize) -> &'a N
    where
        Self: Sized,
    {
        panic!("cannot be called on a vector that is not dense");
    }
}

impl<'a, N: 'a, I: 'a> IntoSparseVecIter<'a, N> for CsVecViewI<'a, N, I>
where
    I: SpIndex,
{
    type IterType = VectorIterator<'a, N, I>;

    fn dim(&self) -> usize {
        self.dim()
    }

    fn into_sparse_vec_iter(self) -> VectorIterator<'a, N, I> {
        self.iter_rbr()
    }
}

impl<'a, N: 'a, I: 'a, IS, DS> IntoSparseVecIter<'a, N>
    for &'a CsVecBase<IS, DS>
where
    I: SpIndex,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [N]>,
{
    type IterType = VectorIterator<'a, N, I>;

    fn dim(&self) -> usize {
        (*self).dim()
    }

    fn into_sparse_vec_iter(self) -> VectorIterator<'a, N, I> {
        self.iter()
    }
}

impl<'a, N: 'a> IntoSparseVecIter<'a, N> for &'a [N] {
    type IterType = Enumerate<Iter<'a, N>>;

    fn dim(&self) -> usize {
        self.len()
    }

    fn into_sparse_vec_iter(self) -> Enumerate<Iter<'a, N>> {
        self.iter().enumerate()
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn index(self, idx: usize) -> &'a N {
        &self[idx]
    }
}

impl<'a, N: 'a> IntoSparseVecIter<'a, N> for &'a Vec<N> {
    type IterType = Enumerate<Iter<'a, N>>;

    fn dim(&self) -> usize {
        self.len()
    }

    fn into_sparse_vec_iter(self) -> Enumerate<Iter<'a, N>> {
        self.iter().enumerate()
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn index(self, idx: usize) -> &'a N {
        &self[idx]
    }
}

impl<'a, N: 'a, S> IntoSparseVecIter<'a, N> for &'a ArrayBase<S, Ix1>
where
    S: ndarray::Data<Elem = N>,
{
    type IterType = Enumerate<ndarray::iter::Iter<'a, N, Ix1>>;

    fn dim(&self) -> usize {
        self.shape()[0]
    }

    fn into_sparse_vec_iter(
        self,
    ) -> Enumerate<ndarray::iter::Iter<'a, N, Ix1>> {
        self.iter().enumerate()
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn index(self, idx: usize) -> &'a N {
        &self[[idx]]
    }
}

/// A trait for types representing dense vectors, useful for
/// defining a fast sparse-dense dot product.
pub trait DenseVector<N> {
    /// The dimension of the vector
    fn dim(&self) -> usize;

    /// Random access to an element in the vector.
    ///
    /// # Panics
    ///
    /// If the index is out of bounds
    fn index(&self, idx: usize) -> &N;
}

impl<'a, N: 'a> DenseVector<N> for &'a [N] {
    fn dim(&self) -> usize {
        self.len()
    }

    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }
}

impl<N> DenseVector<N> for Vec<N> {
    fn dim(&self) -> usize {
        self.len()
    }

    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }
}

impl<'a, N: 'a> DenseVector<N> for &'a Vec<N> {
    fn dim(&self) -> usize {
        self.len()
    }

    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }
}

impl<N, S> DenseVector<N> for ArrayBase<S, Ix1>
where
    S: ndarray::Data<Elem = N>,
{
    fn dim(&self) -> usize {
        self.shape()[0]
    }

    fn index(&self, idx: usize) -> &N {
        &self[[idx]]
    }
}

/// An iterator over the non zeros of either of two vector iterators, ordered,
/// such that the sum of the vectors may be computed
pub struct NnzOrZip<'a, Ite1, Ite2, N1: 'a, N2: 'a>
where
    Ite1: Iterator<Item = (usize, &'a N1)>,
    Ite2: Iterator<Item = (usize, &'a N2)>,
{
    left: Peekable<Ite1>,
    right: Peekable<Ite2>,
    life: PhantomData<(&'a N1, &'a N2)>,
}

#[derive(PartialEq, Debug)]
pub enum NnzEither<'a, N1: 'a, N2: 'a> {
    Both((usize, &'a N1, &'a N2)),
    Left((usize, &'a N1)),
    Right((usize, &'a N2)),
}

fn filter_both_nnz<'a, N: 'a, M: 'a>(
    elem: NnzEither<'a, N, M>,
) -> Option<(usize, &'a N, &'a M)> {
    match elem {
        NnzEither::Both((ind, lval, rval)) => Some((ind, lval, rval)),
        _ => None,
    }
}

impl<'a, Ite1, Ite2, N1: 'a, N2: 'a> Iterator
    for NnzOrZip<'a, Ite1, Ite2, N1, N2>
where
    Ite1: Iterator<Item = (usize, &'a N1)>,
    Ite2: Iterator<Item = (usize, &'a N2)>,
{
    type Item = NnzEither<'a, N1, N2>;

    fn next(&mut self) -> Option<NnzEither<'a, N1, N2>> {
        match (self.left.peek(), self.right.peek()) {
            (None, Some(&(_, _))) => {
                let (rind, rval) = self.right.next().unwrap();
                Some(NnzEither::Right((rind, rval)))
            }
            (Some(&(_, _)), None) => {
                let (lind, lval) = self.left.next().unwrap();
                Some(NnzEither::Left((lind, lval)))
            }
            (None, None) => None,
            (Some(&(lind, _)), Some(&(rind, _))) => {
                if lind < rind {
                    let (lind, lval) = self.left.next().unwrap();
                    Some(NnzEither::Left((lind, lval)))
                } else if rind < lind {
                    let (rind, rval) = self.right.next().unwrap();
                    Some(NnzEither::Right((rind, rval)))
                } else {
                    let (lind, lval) = self.left.next().unwrap();
                    let (_, rval) = self.right.next().unwrap();
                    Some(NnzEither::Both((lind, lval, rval)))
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (left_lower, left_upper) = self.left.size_hint();
        let (right_lower, right_upper) = self.right.size_hint();
        let upper = match (left_upper, right_upper) {
            (Some(x), Some(y)) => Some(x + y),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None,
        };
        (cmp::max(left_lower, right_lower), upper)
    }
}

/// # Methods operating on owning sparse vectors
impl<N, I: SpIndex> CsVecBase<Vec<I>, Vec<N>> {
    /// Create an owning CsVec from vector data.
    ///
    /// # Panics
    ///
    /// - if `indices` and `data` lengths differ
    /// - if the vector contains out of bounds indices
    pub fn new(n: usize, indices: Vec<I>, data: Vec<N>) -> CsVecI<N, I>
    where
        N: Copy,
    {
        Self::try_new(n, indices, data).unwrap()
    }

    /// Try create an owning CsVec from vector data.
    pub fn try_new(
        n: usize,
        mut indices: Vec<I>,
        mut data: Vec<N>,
    ) -> Result<CsVecI<N, I>, SprsError>
    where
        N: Copy,
    {
        let mut buf = Vec::with_capacity(indices.len());
        utils::sort_indices_data_slices(
            &mut indices[..],
            &mut data[..],
            &mut buf,
        );
        let v = CsVecI {
            dim: n,
            indices,
            data,
        };
        v.check_structure().and(Ok(v))
    }

    /// Create an empty CsVec, which can be used for incremental construction
    pub fn empty(dim: usize) -> CsVecI<N, I> {
        CsVecI {
            dim,
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Append an element to the sparse vector. Used for incremental
    /// building of the CsVec. The append should preserve the structure
    /// of the vector, ie the newly added index should be strictly greater
    /// than the last element of indices.
    ///
    /// # Panics
    ///
    /// - Panics if `ind` is lower or equal to the last
    ///   element of `self.indices()`
    /// - Panics if `ind` is greater than `self.dim()`
    pub fn append(&mut self, ind: usize, val: N) {
        match self.indices.last() {
            None => (),
            Some(&last_ind) => {
                assert!(ind > last_ind.index_unchecked(), "unsorted append")
            }
        }
        assert!(ind <= self.dim, "out of bounds index");
        self.indices.push(I::from_usize(ind));
        self.data.push(val);
    }

    /// Reserve `size` additional non-zero values.
    pub fn reserve(&mut self, size: usize) {
        self.indices.reserve(size);
        self.data.reserve(size);
    }

    /// Reserve exactly `exact_size` non-zero values.
    pub fn reserve_exact(&mut self, exact_size: usize) {
        self.indices.reserve_exact(exact_size);
        self.data.reserve_exact(exact_size);
    }

    /// Clear the underlying storage
    pub fn clear(&mut self) {
        self.indices.clear();
        self.data.clear();
    }
}

/// # Common methods of sparse vectors
impl<N, I, IStorage, DStorage> CsVecBase<IStorage, DStorage>
where
    I: SpIndex,
    IStorage: Deref<Target = [I]>,
    DStorage: Deref<Target = [N]>,
{
    /// Get a view of this vector.
    pub fn view(&self) -> CsVecViewI<N, I> {
        CsVecViewI {
            dim: self.dim,
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }

    /// Iterate over the non zero values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sprs::CsVec;
    /// let v = CsVec::new(5, vec![0, 2, 4], vec![1., 2., 3.]);
    /// let mut iter = v.iter();
    /// assert_eq!(iter.next(), Some((0, &1.)));
    /// assert_eq!(iter.next(), Some((2, &2.)));
    /// assert_eq!(iter.next(), Some((4, &3.)));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> VectorIterator<N, I> {
        VectorIterator {
            ind_data: self.indices.iter().zip(self.data.iter()),
        }
    }

    /// Permuted iteration. Not finished
    #[doc(hidden)]
    pub fn iter_perm<'a, 'perm: 'a>(
        &'a self,
        perm: PermViewI<'perm, I>,
    ) -> VectorIteratorPerm<'a, N, I>
    where
        N: 'a,
    {
        VectorIteratorPerm {
            ind_data: self.indices.iter().zip(self.data.iter()),
            perm,
        }
    }

    /// The underlying indices.
    pub fn indices(&self) -> &[I] {
        &self.indices
    }

    /// The underlying non zero values.
    pub fn data(&self) -> &[N] {
        &self.data
    }

    /// Destruct the vector object and recycle its storage containers.
    pub fn into_raw_storage(self) -> (IStorage, DStorage) {
        let Self { indices, data, .. } = self;
        (indices, data)
    }

    /// The dimension of this vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The non zero count of this vector.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Check the sparse structure, namely that:
    /// - indices is sorted
    /// - indices are lower than dims()
    pub fn check_structure(&self) -> Result<(), SprsError> {
        // Make sure indices can be converted to usize
        for i in self.indices.iter() {
            i.index();
        }

        if !self.indices.windows(2).all(|x| x[0] < x[1]) {
            return Err(SprsError::NonSortedIndices);
        }

        if self.dim == 0 && self.indices.len() == 0 && self.data.len() == 0 {
            return Ok(());
        }

        let max_ind = self
            .indices
            .iter()
            .max()
            .unwrap_or(&I::zero())
            .index_unchecked();
        if max_ind >= self.dim {
            return Err(SprsError::IllegalArguments("Out of bounds index"));
        }

        Ok(())
    }

    /// Allocate a new vector equal to this one.
    pub fn to_owned(&self) -> CsVecI<N, I>
    where
        N: Clone,
    {
        CsVecI {
            dim: self.dim,
            indices: self.indices.to_vec(),
            data: self.data.to_vec(),
        }
    }

    /// Clone the vector with another integer type for its indices
    ///
    /// # Panics
    ///
    /// If the indices cannot be represented by the requested integer type.
    pub fn to_other_types<I2>(&self) -> CsVecI<N, I2>
    where
        N: Clone,
        I2: SpIndex,
    {
        let indices = self
            .indices
            .iter()
            .map(|i| I2::from_usize(i.index_unchecked()))
            .collect();
        let data = self.data.iter().cloned().collect();
        CsVecI {
            dim: self.dim,
            indices,
            data,
        }
    }

    /// View this vector as a matrix with only one row.
    pub fn row_view<Iptr: SpIndex>(&self) -> CsMatVecView_<N, I, Iptr> {
        // Safe because we're taking a view into a vector that has
        // necessarily been checked
        let indptr = Array2 {
            data: [
                Iptr::zero(),
                Iptr::from_usize_unchecked(self.indices.len()),
            ],
        };
        CsMatBase {
            storage: CSR,
            nrows: 1,
            ncols: self.dim,
            indptr,
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }

    /// View this vector as a matrix with only one column.
    pub fn col_view<Iptr: SpIndex>(&self) -> CsMatVecView_<N, I, Iptr> {
        // Safe because we're taking a view into a vector that has
        // necessarily been checked
        let indptr = Array2 {
            data: [
                Iptr::zero(),
                Iptr::from_usize_unchecked(self.indices.len()),
            ],
        };
        CsMatBase {
            storage: CSC,
            nrows: self.dim,
            ncols: 1,
            indptr,
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }

    /// Access element at given index, with logarithmic complexity
    pub fn get<'a>(&'a self, index: usize) -> Option<&'a N>
    where
        I: 'a,
    {
        self.view().get_rbr(index)
    }

    /// Find the non-zero index of the requested dimension index,
    /// returning None if no non-zero is present at the requested location.
    ///
    /// Looking for the NnzIndex is done with logarithmic complexity, but
    /// once it is available, the NnzIndex enables retrieving the data with
    /// O(1) complexity.
    pub fn nnz_index(&self, index: usize) -> Option<NnzIndex> {
        self.indices
            .binary_search(&I::from_usize(index))
            .map(|i| NnzIndex(i.index_unchecked()))
            .ok()
    }

    /// Sparse vector dot product. The right-hand-side can be any type
    /// that can be interpreted as a sparse vector (hence sparse vectors, std
    /// vectors and slices, and ndarray's dense vectors work).
    ///
    /// However, even if dense vectors work, it is more performant to use
    /// the [`dot_dense`](struct.CsVecBase.html#method.dot_dense).
    ///
    /// # Panics
    ///
    /// If the dimension of the vectors do not match.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sprs::CsVec;
    /// let v1 = CsVec::new(8, vec![1, 2, 4, 6], vec![1.; 4]);
    /// let v2 = CsVec::new(8, vec![1, 3, 5, 7], vec![2.; 4]);
    /// assert_eq!(2., v1.dot(&v2));
    /// assert_eq!(4., v1.dot(&v1));
    /// assert_eq!(16., v2.dot(&v2));
    /// ```
    pub fn dot<'b, T: IntoSparseVecIter<'b, N>>(&'b self, rhs: T) -> N
    where
        N: 'b + Num + Copy + Sum,
        I: 'b,
        <T as IntoSparseVecIter<'b, N>>::IterType:
            Iterator<Item = (usize, &'b N)>,
        T: Copy, // T is supposed to be a reference type
    {
        assert_eq!(self.dim(), rhs.dim());
        if rhs.is_dense() {
            self.iter()
                .map(|(idx, val)| *val * *rhs.index(idx.index_unchecked()))
                .sum()
        } else {
            let mut lhs_iter = self.iter();
            let mut rhs_iter = rhs.into_sparse_vec_iter();
            let mut sum = N::zero();
            let mut left_nnz = lhs_iter.next();
            let mut right_nnz = rhs_iter.next();
            while left_nnz.is_some() && right_nnz.is_some() {
                let (left_ind, left_val) = left_nnz.unwrap();
                let (right_ind, right_val) = right_nnz.unwrap();
                if left_ind == right_ind {
                    sum = sum + *left_val * *right_val;
                }
                if left_ind <= right_ind {
                    left_nnz = lhs_iter.next();
                }
                if left_ind >= right_ind {
                    right_nnz = rhs_iter.next();
                }
            }
            sum
        }
    }

    /// Sparse-dense vector dot product. The right-hand-side can be any type
    /// that can be interpreted as a dense vector (hence std vectors and
    /// slices, and ndarray's dense vectors work).
    ///
    /// Since the `dot` method can work with the same performance on
    /// dot vectors, the main interest of this method is to enforce at
    /// compile time that the rhs is dense.
    ///
    /// # Panics
    ///
    /// If the dimension of the vectors do not match.
    pub fn dot_dense<T>(&self, rhs: T) -> N
    where
        T: DenseVector<N>,
        N: Num + Copy + Sum,
    {
        assert_eq!(self.dim(), rhs.dim());
        self.iter()
            .map(|(idx, val)| *val * *rhs.index(idx.index_unchecked()))
            .sum()
    }

    /// Compute the squared L2-norm.
    pub fn squared_l2_norm(&self) -> N
    where
        N: Num + Copy + Sum,
    {
        self.data.iter().map(|x| *x * *x).sum()
    }

    /// Compute the L2-norm.
    pub fn l2_norm(&self) -> N
    where
        N: Float + Sum,
    {
        self.squared_l2_norm().sqrt()
    }

    /// Compute the L1-norm.
    pub fn l1_norm(&self) -> N
    where
        N: Signed + Sum,
    {
        self.data.iter().map(|x| x.abs()).sum()
    }

    /// Compute the vector norm for the given order p.
    ///
    /// The norm for vector v is defined as:
    /// - If p = ∞: maxᵢ |vᵢ|
    /// - If p = -∞: minᵢ |vᵢ|
    /// - If p = 0: ∑ᵢ[vᵢ≠0]
    /// - Otherwise: ᵖ√(∑ᵢ|vᵢ|ᵖ)
    pub fn norm(&self, p: N) -> N
    where
        N: Float + Sum,
    {
        let abs_val_iter = self.data.iter().map(|x| x.abs());
        if p.is_infinite() {
            if self.data.is_empty() {
                N::zero()
            } else if p.is_sign_positive() {
                abs_val_iter.fold(N::neg_infinity(), N::max)
            } else {
                abs_val_iter.fold(N::infinity(), N::min)
            }
        } else if p.is_zero() {
            N::from(abs_val_iter.filter(|x| !x.is_zero()).count())
                .expect("Conversion from usize to a Float type should not fail")
        } else {
            abs_val_iter.map(|x| x.powf(p)).sum::<N>().powf(p.powi(-1))
        }
    }

    /// Fill a dense vector with our values
    pub fn scatter(&self, out: &mut [N])
    where
        N: Clone,
    {
        for (ind, val) in self.iter() {
            out[ind] = val.clone();
        }
    }

    /// Transform this vector into a set of (index, value) tuples
    pub fn to_set(&self) -> HashSet<(usize, N)>
    where
        N: Hash + Eq + Clone,
    {
        self.indices()
            .iter()
            .map(|i| i.index_unchecked())
            .zip(self.data.iter().cloned())
            .collect()
    }

    /// Apply a function to each non-zero element, yielding a new matrix
    /// with the same sparsity structure.
    pub fn map<F>(&self, f: F) -> CsVecI<N, I>
    where
        F: FnMut(&N) -> N,
        N: Clone,
    {
        let mut res = self.to_owned();
        res.map_inplace(f);
        res
    }
}

/// # Methods on sparse vectors with mutable access to their data
impl<'a, N, I, IStorage, DStorage> CsVecBase<IStorage, DStorage>
where
    N: 'a,
    I: 'a + SpIndex,
    IStorage: 'a + Deref<Target = [I]>,
    DStorage: DerefMut<Target = [N]>,
{
    /// The underlying non zero values as a mutable slice.
    fn data_mut(&mut self) -> &mut [N] {
        &mut self.data[..]
    }

    pub fn view_mut(&mut self) -> CsVecViewMutI<N, I> {
        CsVecBase {
            dim: self.dim,
            indices: &self.indices[..],
            data: &mut self.data[..],
        }
    }

    /// Access element at given index, with logarithmic complexity
    pub fn get_mut(&mut self, index: usize) -> Option<&mut N> {
        if let Some(NnzIndex(position)) = self.nnz_index(index) {
            Some(&mut self.data[position])
        } else {
            None
        }
    }

    /// Apply a function to each non-zero element, mutating it
    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(&N) -> N,
    {
        for val in &mut self.data[..] {
            *val = f(val);
        }
    }

    /// Mutable iteration over the non-zero values of a sparse vector
    ///
    /// Only the values can be changed, the sparse structure is kept.
    pub fn iter_mut(&mut self) -> VectorIteratorMut<N, I> {
        VectorIteratorMut {
            ind_data: self.indices.iter().zip(self.data.iter_mut()),
        }
    }

    /// Divides the vector by its own L2-norm.
    ///
    /// Zero vector is left unchanged.
    pub fn unit_normalize(&mut self)
    where
        N: Float + Sum,
    {
        let norm_sq = self.squared_l2_norm();
        if norm_sq > N::zero() {
            let norm = norm_sq.sqrt();
            self.map_inplace(|x| *x / norm);
        }
    }
}

/// # Methods propagating the lifetime of a `CsVecViewI`.
impl<'a, N: 'a, I: 'a + SpIndex> CsVecBase<&'a [I], &'a [N]> {
    /// Create a borrowed CsVec over slice data.
    pub fn new_view(
        n: usize,
        indices: &'a [I],
        data: &'a [N],
    ) -> Result<CsVecViewI<'a, N, I>, SprsError> {
        let v = CsVecViewI {
            dim: n,
            indices,
            data,
        };
        v.check_structure().and(Ok(v))
    }

    /// Access element at given index, with logarithmic complexity
    ///
    /// Re-borrowing version of `at()`.
    pub fn get_rbr(&self, index: usize) -> Option<&'a N> {
        self.nnz_index(index)
            .map(|NnzIndex(position)| &self.data[position])
    }

    /// Re-borrowing version of `iter()`. Namely, the iterator's lifetime
    /// will be bound to the lifetime of the underlying slices instead
    /// of being bound to the lifetime of the borrow.
    fn iter_rbr(&self) -> VectorIterator<'a, N, I> {
        VectorIterator {
            ind_data: self.indices.iter().zip(self.data.iter()),
        }
    }

    /// Create a borrowed CsVec over slice data without checking the structure
    /// This is unsafe because algorithms are free to assume
    /// that properties guaranteed by check_structure are enforced.
    /// For instance, non out-of-bounds indices can be relied upon to
    /// perform unchecked slice access.
    pub unsafe fn new_view_raw(
        n: usize,
        nnz: usize,
        indices: *const I,
        data: *const N,
    ) -> CsVecViewI<'a, N, I> {
        CsVecViewI {
            dim: n,
            indices: slice::from_raw_parts(indices, nnz),
            data: slice::from_raw_parts(data, nnz),
        }
    }
}

/// # Methods propagating the lifetome of a `CsVecViewMutI`.
impl<'a, N, I> CsVecBase<&'a [I], &'a mut [N]>
where
    N: 'a,
    I: 'a + SpIndex,
{
    /// Create a borrowed CsVec over slice data without checking the structure
    /// This is unsafe because algorithms are free to assume
    /// that properties guaranteed by check_structure are enforced, and
    /// because the lifetime of the pointers is unconstrained.
    /// For instance, non out-of-bounds indices can be relied upon to
    /// perform unchecked slice access.
    /// For safety, lifetime of the resulting vector should match the lifetime
    /// of the input pointers.
    pub unsafe fn new_view_mut_raw(
        n: usize,
        nnz: usize,
        indices: *const I,
        data: *mut N,
    ) -> CsVecViewMutI<'a, N, I> {
        CsVecBase {
            dim: n,
            indices: slice::from_raw_parts(indices, nnz),
            data: slice::from_raw_parts_mut(data, nnz),
        }
    }
}

impl<'a, 'b, N, I, Iptr, IS1, DS1, IpS2, IS2, DS2>
    Mul<&'b CsMatBase<N, I, IpS2, IS2, DS2, Iptr>> for &'a CsVecBase<IS1, DS1>
where
    N: 'a + Copy + Num + Default,
    I: 'a + SpIndex,
    Iptr: 'a + SpIndex,
    IS1: 'a + Deref<Target = [I]>,
    DS1: 'a + Deref<Target = [N]>,
    IpS2: 'b + Deref<Target = [Iptr]>,
    IS2: 'b + Deref<Target = [I]>,
    DS2: 'b + Deref<Target = [N]>,
{
    type Output = CsVecI<N, I>;

    fn mul(self, rhs: &CsMatBase<N, I, IpS2, IS2, DS2, Iptr>) -> CsVecI<N, I> {
        (&self.row_view() * rhs).outer_view(0).unwrap().to_owned()
    }
}

impl<'a, 'b, N, I, Iptr, IpS1, IS1, DS1, IS2, DS2> Mul<&'b CsVecBase<IS2, DS2>>
    for &'a CsMatBase<N, I, IpS1, IS1, DS1, Iptr>
where
    N: Copy + Num + Default + Sum,
    I: SpIndex,
    Iptr: SpIndex,
    IpS1: Deref<Target = [Iptr]>,
    IS1: Deref<Target = [I]>,
    DS1: Deref<Target = [N]>,
    IS2: Deref<Target = [I]>,
    DS2: Deref<Target = [N]>,
{
    type Output = CsVecI<N, I>;

    fn mul(self, rhs: &CsVecBase<IS2, DS2>) -> CsVecI<N, I> {
        if self.is_csr() {
            prod::csr_mul_csvec(self.view(), rhs.view())
        } else {
            (self * &rhs.col_view()).outer_view(0).unwrap().to_owned()
        }
    }
}

impl<N, I, IS1, DS1, IS2, DS2> Add<CsVecBase<IS2, DS2>> for CsVecBase<IS1, DS1>
where
    N: Copy + Num,
    I: SpIndex,
    IS1: Deref<Target = [I]>,
    DS1: Deref<Target = [N]>,
    IS2: Deref<Target = [I]>,
    DS2: Deref<Target = [N]>,
{
    type Output = CsVecI<N, I>;

    fn add(self, rhs: CsVecBase<IS2, DS2>) -> CsVecI<N, I> {
        &self + &rhs
    }
}

impl<'a, N, I, IS1, DS1, IS2, DS2> Add<&'a CsVecBase<IS2, DS2>>
    for CsVecBase<IS1, DS1>
where
    N: Copy + Num,
    I: SpIndex,
    IS1: Deref<Target = [I]>,
    DS1: Deref<Target = [N]>,
    IS2: Deref<Target = [I]>,
    DS2: Deref<Target = [N]>,
{
    type Output = CsVecI<N, I>;

    fn add(self, rhs: &CsVecBase<IS2, DS2>) -> CsVecI<N, I> {
        &self + rhs
    }
}

impl<'a, N, I, IS1, DS1, IS2, DS2> Add<CsVecBase<IS2, DS2>>
    for &'a CsVecBase<IS1, DS1>
where
    N: Copy + Num,
    I: SpIndex,
    IS1: Deref<Target = [I]>,
    DS1: Deref<Target = [N]>,
    IS2: Deref<Target = [I]>,
    DS2: Deref<Target = [N]>,
{
    type Output = CsVecI<N, I>;

    fn add(self, rhs: CsVecBase<IS2, DS2>) -> CsVecI<N, I> {
        self + &rhs
    }
}

impl<'a, 'b, N, I, IS1, DS1, IS2, DS2> Add<&'b CsVecBase<IS2, DS2>>
    for &'a CsVecBase<IS1, DS1>
where
    N: Copy + Num,
    I: SpIndex,
    IS1: Deref<Target = [I]>,
    DS1: Deref<Target = [N]>,
    IS2: Deref<Target = [I]>,
    DS2: Deref<Target = [N]>,
{
    type Output = CsVecI<N, I>;

    fn add(self, rhs: &CsVecBase<IS2, DS2>) -> CsVecI<N, I> {
        binop::csvec_binop(self.view(), rhs.view(), |&x, &y| x + y).unwrap()
    }
}

impl<'a, 'b, N, IS1, DS1, IS2, DS2> Sub<&'b CsVecBase<IS2, DS2>>
    for &'a CsVecBase<IS1, DS1>
where
    N: Copy + Num,
    IS1: Deref<Target = [usize]>,
    DS1: Deref<Target = [N]>,
    IS2: Deref<Target = [usize]>,
    DS2: Deref<Target = [N]>,
{
    type Output = CsVec<N>;

    fn sub(self, rhs: &CsVecBase<IS2, DS2>) -> CsVec<N> {
        binop::csvec_binop(self.view(), rhs.view(), |&x, &y| x - y).unwrap()
    }
}

impl<N: Num + Copy + Neg<Output = N>, I: SpIndex> Neg for CsVecI<N, I> {
    type Output = CsVecI<N, I>;

    fn neg(mut self) -> CsVecI<N, I> {
        for value in &mut self.data {
            *value = -*value;
        }
        self
    }
}

impl<N, IS, DS> Index<usize> for CsVecBase<IS, DS>
where
    IS: Deref<Target = [usize]>,
    DS: Deref<Target = [N]>,
{
    type Output = N;

    fn index(&self, index: usize) -> &N {
        self.get(index).unwrap()
    }
}

impl<N, IS, DS> IndexMut<usize> for CsVecBase<IS, DS>
where
    IS: Deref<Target = [usize]>,
    DS: DerefMut<Target = [N]>,
{
    fn index_mut(&mut self, index: usize) -> &mut N {
        self.get_mut(index).unwrap()
    }
}

impl<N, IS, DS> Index<NnzIndex> for CsVecBase<IS, DS>
where
    IS: Deref<Target = [usize]>,
    DS: Deref<Target = [N]>,
{
    type Output = N;

    fn index(&self, index: NnzIndex) -> &N {
        let NnzIndex(i) = index;
        self.data().get(i).unwrap()
    }
}

impl<N, IS, DS> IndexMut<NnzIndex> for CsVecBase<IS, DS>
where
    IS: Deref<Target = [usize]>,
    DS: DerefMut<Target = [N]>,
{
    fn index_mut(&mut self, index: NnzIndex) -> &mut N {
        let NnzIndex(i) = index;
        self.data_mut().get_mut(i).unwrap()
    }
}

impl<N: Num + Copy, I: SpIndex> Zero for CsVecI<N, I> {
    fn zero() -> CsVecI<N, I> {
        CsVecI::new(0, vec![], vec![])
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.is_zero())
    }
}

#[cfg(feature = "alga")]
mod alga_impls {
    use super::*;
    use alga::general::*;

    impl<N: Clone + Copy + Num, I: Clone + SpIndex> AbstractMagma<Additive>
        for CsVecI<N, I>
    {
        fn operate(&self, right: &CsVecI<N, I>) -> CsVecI<N, I> {
            self + right
        }
    }

    impl<N: Copy + Num, I: SpIndex> Identity<Additive> for CsVecI<N, I> {
        fn identity() -> CsVecI<N, I> {
            CsVecI::zero()
        }
    }

    impl<N: Copy + Num, I: SpIndex> AbstractSemigroup<Additive> for CsVecI<N, I> {}

    impl<N: Copy + Num, I: SpIndex> AbstractMonoid<Additive> for CsVecI<N, I> {}

    impl<N, I> TwoSidedInverse<Additive> for CsVecI<N, I>
    where
        N: Clone + Neg<Output = N> + Copy + Num,
        I: SpIndex,
    {
        fn two_sided_inverse(&self) -> CsVecI<N, I> {
            CsVecBase {
                data: self.data.iter().map(|x| -*x).collect(),
                indices: self.indices.clone(),
                dim: self.dim,
            }
        }
    }

    impl<N: Copy + Num + Neg<Output = N>, I: SpIndex>
        AbstractQuasigroup<Additive> for CsVecI<N, I>
    {
    }

    impl<N: Copy + Num + Neg<Output = N>, I: SpIndex> AbstractLoop<Additive>
        for CsVecI<N, I>
    {
    }

    impl<N: Copy + Num + Neg<Output = N>, I: SpIndex> AbstractGroup<Additive>
        for CsVecI<N, I>
    {
    }

    impl<N: Copy + Num + Neg<Output = N>, I: SpIndex>
        AbstractGroupAbelian<Additive> for CsVecI<N, I>
    {
    }

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn additive_operator_is_addition() {
            let a = CsVec::new(2, vec![0], vec![2.]);
            let b = CsVec::new(2, vec![0], vec![3.]);
            assert_eq!(AbstractMagma::<Additive>::operate(&a, &b), &a + &b);
        }

        #[test]
        fn additive_identity_is_zero() {
            assert_eq!(CsVec::<f64>::zero(), Identity::<Additive>::identity());
        }

        #[test]
        fn additive_inverse_is_negated() {
            let vector = CsVec::new(2, vec![0], vec![2.]);
            assert_eq!(
                -vector.clone(),
                TwoSidedInverse::<Additive>::two_sided_inverse(&vector)
            );
        }
    }
}

#[cfg(test)]
mod test {
    use super::SparseIterTools;
    use ndarray::Array;
    use num_traits::Zero;
    use sparse::{CsVec, CsVecI};

    fn test_vec1() -> CsVec<f64> {
        let n = 8;
        let indices = vec![0, 1, 4, 5, 7];
        let data = vec![0., 1., 4., 5., 7.];

        return CsVec::new(n, indices, data);
    }

    fn test_vec2() -> CsVecI<f64, usize> {
        let n = 8;
        let indices = vec![0, 2, 4, 6, 7];
        let data = vec![0.5, 2.5, 4.5, 6.5, 7.5];

        return CsVecI::new(n, indices, data);
    }

    #[test]
    fn test_copy() {
        let v = test_vec1();
        let view1 = v.view();
        let view2 = view1; // this shouldn't move
        assert_eq!(view1, view2);
    }

    #[test]
    fn test_nnz_zip_iter() {
        let vec1 = test_vec1();
        let vec2 = test_vec2();
        let mut iter = vec1.iter().nnz_zip(vec2.iter());
        assert_eq!(iter.next().unwrap(), (0, &0., &0.5));
        assert_eq!(iter.next().unwrap(), (4, &4., &4.5));
        assert_eq!(iter.next().unwrap(), (7, &7., &7.5));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_nnz_or_zip_iter() {
        use super::NnzEither::*;
        let vec1 = test_vec1();
        let vec2 = test_vec2();
        let mut iter = vec1.iter().nnz_or_zip(vec2.iter());
        assert_eq!(iter.next().unwrap(), Both((0, &0., &0.5)));
        assert_eq!(iter.next().unwrap(), Left((1, &1.)));
        assert_eq!(iter.next().unwrap(), Right((2, &2.5)));
        assert_eq!(iter.next().unwrap(), Both((4, &4., &4.5)));
        assert_eq!(iter.next().unwrap(), Left((5, &5.)));
        assert_eq!(iter.next().unwrap(), Right((6, &6.5)));
        assert_eq!(iter.next().unwrap(), Both((7, &7., &7.5)));
    }

    #[test]
    fn dot_product() {
        let vec1 = CsVec::new(8, vec![0, 2, 4, 6], vec![1.; 4]);
        let vec2 = CsVec::new(8, vec![1, 3, 5, 7], vec![2.; 4]);
        let vec3 = CsVec::new(8, vec![1, 2, 5, 6], vec![3.; 4]);

        assert_eq!(0., vec1.dot(&vec2));
        assert_eq!(4., vec1.dot(&vec1));
        assert_eq!(16., vec2.dot(&vec2));
        assert_eq!(6., vec1.dot(&vec3));
        assert_eq!(12., vec2.dot(vec3.view()));

        let dense_vec = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        {
            let slice = &dense_vec[..];
            assert_eq!(16., vec1.dot(&dense_vec));
            assert_eq!(16., vec1.dot(slice));
            assert_eq!(16., vec1.dot_dense(slice));
            assert_eq!(16., vec1.dot_dense(&dense_vec));
        }
        assert_eq!(16., vec1.dot_dense(dense_vec));

        let ndarray_vec = Array::linspace(1., 8., 8);
        assert_eq!(16., vec1.dot(&ndarray_vec));
        assert_eq!(16., vec1.dot_dense(ndarray_vec.view()));
    }

    #[test]
    #[should_panic]
    fn dot_product_panics() {
        let vec1 = CsVec::new(8, vec![0, 2, 4, 6], vec![1.; 4]);
        let vec2 = CsVec::new(9, vec![1, 3, 5, 7], vec![2.; 4]);
        vec1.dot(&vec2);
    }

    #[test]
    #[should_panic]
    fn dot_product_panics2() {
        let vec1 = CsVec::new(8, vec![0, 2, 4, 6], vec![1.; 4]);
        let dense_vec = vec![0., 1., 2., 3., 4., 5., 6., 7., 8.];
        vec1.dot(&dense_vec);
    }

    #[test]
    fn squared_l2_norm() {
        // Should work with both float and integer data

        let v = CsVec::new(0, Vec::<usize>::new(), Vec::<i32>::new());
        assert_eq!(0, v.squared_l2_norm());

        let v = CsVec::new(0, Vec::<usize>::new(), Vec::<f32>::new());
        assert_eq!(0., v.squared_l2_norm());

        let v = CsVec::new(8, vec![0, 1, 4, 5, 7], vec![0, 1, 4, 5, 7]);
        assert_eq!(v.dot(&v), v.squared_l2_norm());

        let v = CsVec::new(8, vec![0, 1, 4, 5, 7], vec![0., 1., 4., 5., 7.]);
        assert_eq!(v.dot(&v), v.squared_l2_norm());
    }

    #[test]
    fn l2_norm() {
        let v = CsVec::new(0, Vec::<usize>::new(), Vec::<f32>::new());
        assert_eq!(0., v.l2_norm());

        let v = test_vec1();
        assert_eq!(v.dot(&v).sqrt(), v.l2_norm());
    }

    #[test]
    fn unit_normalize() {
        let mut v = CsVec::new(0, Vec::<usize>::new(), Vec::<f32>::new());
        v.unit_normalize();
        assert_eq!(0, v.nnz());
        assert!(v.indices.is_empty());
        assert!(v.data.is_empty());

        let mut v = CsVec::new(8, vec![1, 3, 5], vec![0., 0., 0.]);
        v.unit_normalize();
        assert_eq!(3, v.nnz());
        assert!(v.data.iter().all(|x| x.is_zero()));

        let mut v =
            CsVec::new(8, vec![0, 1, 4, 5, 7], vec![0., 1., 4., 5., 7.]);
        v.unit_normalize();
        let norm = (1f32 + 4. * 4. + 5. * 5. + 7. * 7.).sqrt();
        assert_eq!(
            vec![0., 1. / norm, 4. / norm, 5. / norm, 7. / norm],
            v.data
        );
        assert!((v.l2_norm() - 1.).abs() < 1e-5);
    }

    #[test]
    fn l1_norm() {
        let v = CsVec::new(0, Vec::<usize>::new(), Vec::<f32>::new());
        assert_eq!(0., v.l1_norm());

        let v = CsVec::new(8, vec![0, 1, 4, 5, 7], vec![0, -1, 4, -5, 7]);
        assert_eq!(1 + 4 + 5 + 7, v.l1_norm());
    }

    #[test]
    fn norm() {
        let v = CsVec::new(0, Vec::<usize>::new(), Vec::<f32>::new());
        assert_eq!(0., v.norm(std::f32::INFINITY)); // Here we choose the same behavior as Eigen
        assert_eq!(0., v.norm(0.));
        assert_eq!(0., v.norm(5.));

        let v = CsVec::new(8, vec![0, 1, 4, 5, 7], vec![0., 1., -4., 5., -7.]);
        assert_eq!(7., v.norm(std::f32::INFINITY));
        assert_eq!(0., v.norm(std::f32::NEG_INFINITY));
        assert_eq!(4., v.norm(0.));
        assert_eq!(v.l1_norm(), v.norm(1.));
        assert_eq!(v.l2_norm(), v.norm(2.));
    }

    #[test]
    fn nnz_index() {
        let vec = CsVec::new(8, vec![0, 2, 4, 6], vec![1.; 4]);
        assert_eq!(vec.nnz_index(1), None);
        assert_eq!(vec.nnz_index(9), None);
        assert_eq!(vec.nnz_index(0), Some(super::NnzIndex(0)));
        assert_eq!(vec.nnz_index(4), Some(super::NnzIndex(2)));

        let index = vec.nnz_index(2).unwrap();

        assert_eq!(vec[index], 1.);
        let mut vec = vec;
        vec[index] = 2.;
        assert_eq!(vec[index], 2.);
    }

    #[test]
    fn get_mut() {
        let mut vec = CsVec::new(8, vec![0, 2, 4, 6], vec![1.; 4]);

        *vec.get_mut(4).unwrap() = 2.;

        let expected = CsVec::new(8, vec![0, 2, 4, 6], vec![1., 1., 2., 1.]);

        assert_eq!(vec, expected);

        vec[6] = 3.;

        let expected = CsVec::new(8, vec![0, 2, 4, 6], vec![1., 1., 2., 3.]);

        assert_eq!(vec, expected);
    }

    #[test]
    fn indexing() {
        let vec = CsVec::new(8, vec![0, 2, 4, 6], vec![1., 2., 3., 4.]);
        assert_eq!(vec[0], 1.);
        assert_eq!(vec[2], 2.);
        assert_eq!(vec[4], 3.);
        assert_eq!(vec[6], 4.);
    }

    #[test]
    fn map_inplace() {
        let mut vec = CsVec::new(8, vec![0, 2, 4, 6], vec![1., 2., 3., 4.]);
        vec.map_inplace(|&x| x + 1.);
        let expected = CsVec::new(8, vec![0, 2, 4, 6], vec![2., 3., 4., 5.]);
        assert_eq!(vec, expected);
    }

    #[test]
    fn map() {
        let vec = CsVec::new(8, vec![0, 2, 4, 6], vec![1., 2., 3., 4.]);
        let res = vec.map(|&x| x * 2.);
        let expected = CsVec::new(8, vec![0, 2, 4, 6], vec![2., 4., 6., 8.]);
        assert_eq!(res, expected);
    }

    #[test]
    fn iter_mut() {
        let mut vec = CsVec::new(8, vec![0, 2, 4, 6], vec![1., 2., 3., 4.]);
        for (ind, val) in vec.iter_mut() {
            if ind == 2 {
                *val += 1.;
            } else {
                *val *= 2.;
            }
        }
        let expected = CsVec::new(8, vec![0, 2, 4, 6], vec![2., 3., 6., 8.]);
        assert_eq!(vec, expected);
    }

    #[test]
    fn adds_vectors_by_value() {
        let (a, b, expected_sum) = addition_sample();
        assert_eq!(expected_sum, a + b);
    }

    #[test]
    fn adds_vectors_by_left_value_and_right_reference() {
        let (a, b, expected_sum) = addition_sample();
        assert_eq!(expected_sum, a + &b);
    }

    #[test]
    fn adds_vectors_by_left_reference_and_right_value() {
        let (a, b, expected_sum) = addition_sample();
        assert_eq!(expected_sum, &a + b);
    }

    #[test]
    fn adds_vectors_by_reference() {
        let (a, b, expected_sum) = addition_sample();
        assert_eq!(expected_sum, &a + &b);
    }

    fn addition_sample() -> (CsVec<f64>, CsVec<f64>, CsVec<f64>) {
        let dim = 8;
        let a = CsVec::new(dim, vec![0, 3, 5, 7], vec![2., -3., 7., -1.]);
        let b = CsVec::new(dim, vec![1, 3, 4, 5], vec![4., 2., -3., 1.]);
        let expected_sum = CsVec::new(
            dim,
            vec![0, 1, 3, 4, 5, 7],
            vec![2., 4., -1., -3., 8., -1.],
        );
        (a, b, expected_sum)
    }

    #[test]
    fn negates_vectors() {
        let vector = CsVec::new(4, vec![0, 3], vec![2., -3.]);
        let negated = CsVec::new(4, vec![0, 3], vec![-2., 3.]);
        assert_eq!(-vector, negated);
    }

    #[test]
    fn can_construct_zero_sized_vectors() {
        CsVec::<f64>::new(0, vec![], vec![]);
    }

    #[test]
    fn zero_element_vanishes_when_added() {
        let zero = CsVec::<f64>::zero();
        let vector = CsVec::new(3, vec![0, 2], vec![1., 2.]);
        assert_eq!(&vector + &zero, vector);
    }

    #[test]
    fn zero_element_is_identified_as_zero() {
        assert!(CsVec::<f32>::zero().is_zero());
    }

    #[test]
    fn larger_zero_vector_is_identified_as_zero() {
        let vector = CsVec::new(3, vec![1, 2], vec![0., 0.]);
        assert!(vector.is_zero());
    }
}
