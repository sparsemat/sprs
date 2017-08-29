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

use std::iter::{Zip, Peekable, FilterMap, IntoIterator, Enumerate};
use std::ops::{Deref, DerefMut, Mul, Add, Sub, Index, IndexMut};
use std::convert::AsRef;
use std::cmp;
use std::slice::{self, Iter, IterMut};
use std::collections::HashSet;
use std::hash::Hash;
use std::marker::PhantomData;
use ndarray::{self, ArrayBase};
use ::{Ix1};

use num_traits::Num;

use indexing::SpIndex;
use array_backend::Array2;
use sparse::permutation::PermViewI;
use sparse::{prod, binop};
use sparse::utils;
use sparse::prelude::*;
use sparse::csmat::CompressedStorage::{CSR, CSC};
use errors::SprsError;

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

impl<N, IS, DS: Deref<Target=[N]>> VecDim<N> for CsVecBase<IS, DS> {
    fn dim(&self) -> usize {
        self.dim
    }
}

impl<N, T: ?Sized> VecDim<N> for T where T: AsRef<[N]> {
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


impl <'a, N: 'a, I: 'a + SpIndex>
Iterator
for VectorIterator<'a, N, I> {
    type Item = (usize, &'a N);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.ind_data.next() {
            None => None,
            Some((inner_ind, data)) => Some((inner_ind.index(), data))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ind_data.size_hint()
    }
}

impl <'a, N: 'a, I: 'a + SpIndex>
Iterator
for VectorIteratorPerm<'a, N, I> {
    type Item = (usize, &'a N);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.ind_data.next() {
            None => None,
            Some((inner_ind, data)) => Some(
                (self.perm.at(inner_ind.index()), data))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ind_data.size_hint()
    }
}

impl <'a, N: 'a, I: 'a + SpIndex>
Iterator
for VectorIteratorMut<'a, N, I> {
    type Item = (usize, &'a mut N);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.ind_data.next() {
            None => None,
            Some((inner_ind, data)) => Some((inner_ind.index(), data))
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
    fn nnz_or_zip<'a, I, N1, N2>(self, other: I)
    -> NnzOrZip<'a, Self, I::IntoIter, N1, N2>
    where Self: Iterator<Item=(usize, &'a N1)> + Sized,
          I: IntoIterator<Item=(usize, &'a N2)> {
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
    fn nnz_zip<'a, I, N1, N2>(self, other: I)
    -> FilterMap<NnzOrZip<'a, Self, I::IntoIter, N1, N2>,
                 fn(NnzEither<'a, N1,N2>) -> Option<(usize, &'a N1, &'a N2)>>
    where Self: Iterator<Item=(usize, &'a N1)> + Sized,
          I: IntoIterator<Item=(usize, &'a N2)> {
        let nnz_or_iter = NnzOrZip {
            left: self.peekable(),
            right: other.into_iter().peekable(),
            life: PhantomData,
        };
        nnz_or_iter.filter_map(filter_both_nnz)
    }
}

impl<T: Iterator> SparseIterTools for Enumerate<T> {
}

impl<'a, N: 'a, I: 'a + SpIndex>
SparseIterTools
for VectorIterator<'a, N, I> {
}

/// Trait for types that can be iterated as sparse vectors
pub trait IntoSparseVecIter<N> {

    type IterType;

    /// Transform self into an iterator that yields (usize, &N) tuples
    /// where the usize is the index of the value in the sparse vector.
    /// The indices should be sorted.
    fn into_sparse_vec_iter(self) -> <Self as IntoSparseVecIter<N>>::IterType
    where <Self as IntoSparseVecIter<N>>::IterType: Iterator<Item=(usize, N)>;

    /// The dimension of the vector
    fn dim(&self) -> usize;
}

impl<'a, N: 'a, I: 'a> IntoSparseVecIter<&'a N> for CsVecViewI<'a, N, I>
where I: SpIndex,
{
    type IterType = VectorIterator<'a, N, I>;

    fn dim(&self) -> usize {
        self.dim()
    }

    fn into_sparse_vec_iter(self) -> VectorIterator<'a, N, I> {
        self.iter_rbr()
    }
}

impl<'a, N: 'a, I: 'a, IS, DS>
IntoSparseVecIter<&'a N>
for &'a CsVecBase<IS, DS>
where I: SpIndex,
      IS: Deref<Target=[I]>,
      DS: Deref<Target=[N]>
{
    type IterType = VectorIterator<'a, N, I>;

    fn dim(&self) -> usize {
        (*self).dim()
    }

    fn into_sparse_vec_iter(self) -> VectorIterator<'a, N, I> {
        self.iter()
    }
}

impl<'a, N: 'a> IntoSparseVecIter<&'a N> for &'a [N] {
    type IterType = Enumerate<Iter<'a, N>>;

    fn dim(&self) -> usize {
        self.len()
    }

    fn into_sparse_vec_iter(self) -> Enumerate<Iter<'a, N>> {
        self.into_iter().enumerate()
    }
}

impl<'a, N: 'a> IntoSparseVecIter<&'a N> for &'a Vec<N> {
    type IterType = Enumerate<Iter<'a, N>>;

    fn dim(&self) -> usize {
        self.len()
    }

    fn into_sparse_vec_iter(self) -> Enumerate<Iter<'a, N>> {
        self.into_iter().enumerate()
    }
}

impl<'a, N: 'a, S> IntoSparseVecIter<&'a N> for &'a ArrayBase<S, Ix1>
where S: ndarray::Data<Elem=N>
{
    type IterType = Enumerate<ndarray::iter::Iter<'a, N, Ix1>>;

    fn dim(&self) -> usize {
        self.shape()[0]
    }

    fn into_sparse_vec_iter(self) -> Enumerate<ndarray::iter::Iter<'a, N, Ix1>> {
        self.iter().enumerate()
    }
}

/// An iterator over the non zeros of either of two vector iterators, ordered,
/// such that the sum of the vectors may be computed
pub struct NnzOrZip<'a, Ite1, Ite2, N1: 'a, N2: 'a>
where Ite1: Iterator<Item=(usize, &'a N1)>,
      Ite2: Iterator<Item=(usize, &'a N2)> {
    left: Peekable<Ite1>,
    right: Peekable<Ite2>,
    life: PhantomData<(&'a N1, &'a N2)>,
}

#[derive(PartialEq, Debug)]
pub enum NnzEither<'a, N1: 'a, N2: 'a> {
    Both((usize, &'a N1, &'a N2)),
    Left((usize, &'a N1)),
    Right((usize, &'a N2))
}

fn filter_both_nnz<'a, N: 'a, M: 'a>(elem: NnzEither<'a, N, M>)
-> Option<(usize, &'a N, &'a M)> {
    match elem {
        NnzEither::Both((ind, lval, rval)) => Some((ind, lval, rval)),
        _ => None
    }
}

impl <'a, Ite1, Ite2, N1: 'a, N2: 'a>
Iterator
for NnzOrZip<'a, Ite1, Ite2, N1, N2>
where Ite1: Iterator<Item=(usize, &'a N1)>,
      Ite2: Iterator<Item=(usize, &'a N2)> {
    type Item = NnzEither<'a, N1, N2>;

    fn next(&mut self) -> Option<(NnzEither<'a, N1, N2>)> {
        match (self.left.peek(), self.right.peek()) {
            (None, Some(&(_, _))) => {
                let (rind, rval) = self.right.next().unwrap();
                Some(NnzEither::Right((rind, rval)))
            }
            (Some(&(_,_)), None) => {
                let (lind, lval) = self.left.next().unwrap();
                Some(NnzEither::Left((lind, lval)))
            }
            (None, None) => None,
            (Some(&(lind, _)), Some(&(rind, _))) => {
                if lind < rind {
                    let (lind, lval) = self.left.next().unwrap();
                    Some(NnzEither::Left((lind, lval)))
                }
                else if rind < lind {
                    let (rind, rval) = self.right.next().unwrap();
                    Some(NnzEither::Right((rind, rval)))
                }
                else {
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
            (None, None) => None
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
    pub fn new(n: usize,
               mut indices: Vec<I>,
               mut data: Vec<N>
              ) -> CsVecI<N, I>
    where N: Copy
    {
        let mut buf = Vec::with_capacity(indices.len());
        utils::sort_indices_data_slices(&mut indices[..],
                                        &mut data[..],
                                        &mut buf);
        let v = CsVecI {
            dim: n,
            indices: indices,
            data: data
        };
        v.check_structure().and(Ok(v)).unwrap()
    }

    /// Create an empty CsVec, which can be used for incremental construction
    pub fn empty(dim: usize) -> CsVecI<N, I> {
        CsVecI {
            dim: dim,
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
                assert!(ind > last_ind.index(), "unsorted append")
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
where I: SpIndex,
      IStorage: Deref<Target=[I]>,
      DStorage: Deref<Target=[N]> {

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
    pub fn iter_perm<'a, 'perm: 'a>(&'a self,
                                    perm: PermViewI<'perm, I>)
    -> VectorIteratorPerm<'a, N, I>
    where N: 'a
    {
        VectorIteratorPerm {
            ind_data: self.indices.iter().zip(self.data.iter()),
            perm: perm
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
        if ! self.indices.windows(2).all(|x| x[0] < x[1]) {
            return Err(SprsError::NonSortedIndices);
        }

        let max_ind = self.indices.iter().max().unwrap_or(&I::zero()).index();
        if max_ind >= self.dim {
            panic!("Out of bounds index");
        }

        Ok(())
    }

    /// Allocate a new vector equal to this one.
    pub fn to_owned(&self) -> CsVecI<N, I>
    where N: Clone
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
    pub fn to_other_idx_type<I2>(&self) -> CsVecI<N, I2>
    where N: Clone,
          I2: SpIndex,
    {
        let indices = self.indices.iter()
                                  .map(|i| I2::from_usize(i.index()))
                                  .collect();
        CsVecI {
            dim: self.dim,
            indices: indices,
            data: self.data.to_vec(),
        }
    }

    /// View this vector as a matrix with only one row.
    pub fn row_view(&self) -> CsMatVecView_<N, I> {
        // Safe because we're taking a view into a vector that has
        // necessarily been checked
        let indptr = Array2 {
            data: [I::zero(), I::from_usize(self.indices.len())],
        };
        CsMatBase {
            storage: CSR,
            nrows: 1,
            ncols: self.dim,
            indptr: indptr,
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }

    /// View this vector as a matrix with only one column.
    pub fn col_view(&self) -> CsMatVecView_<N, I> {
        // Safe because we're taking a view into a vector that has
        // necessarily been checked
        let indptr = Array2 {
            data: [I::zero(), I::from_usize(self.indices.len())],
        };
        CsMatBase {
            storage: CSC,
            nrows: self.dim,
            ncols: 1,
            indptr: indptr,
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }

    /// Access element at given index, with logarithmic complexity
    pub fn get<'a>(&'a self, index: usize) -> Option<&'a N>
    where I: 'a
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
        self.indices.binary_search(&I::from_usize(index))
                    .map(|i| NnzIndex(i.index()))
                    .ok()
    }

    /// Sparse vector dot product. The right-hand-side can be any type
    /// that can be interpreted as a sparse vector (hence sparse vectors, std
    /// vectors and slices, and ndarray's dense vectors work).
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
    pub fn dot<'b, T: IntoSparseVecIter<&'b N>>(&'b self, rhs: T) -> N
    where N: 'b + Num + Copy,
          I: 'b,
          <T as IntoSparseVecIter<&'b N>>::IterType: Iterator<Item=(usize, &'b N)>
    {
        assert_eq!(self.dim(), rhs.dim());
        self.iter().nnz_zip(rhs.into_sparse_vec_iter())
                   .map(|(_, &lval, &rval)| lval * rval)
                   .fold(N::zero(), |x, y| x + y)
    }

    /// Fill a dense vector with our values
    pub fn scatter(&self, out: &mut [N])
    where N: Clone {
        for (ind, val) in self.iter() {
            out[ind] = val.clone();
        }
    }

    /// Transform this vector into a set of (index, value) tuples
    pub fn to_set(self) -> HashSet<(usize, N)>
    where N: Hash + Eq + Clone {
        self.indices().iter().map(|i| i.index())
            .zip(self.data.iter().cloned())
            .collect()
    }

    /// Apply a function to each non-zero element, yielding a new matrix
    /// with the same sparsity structure.
    pub fn map<F>(&self, f: F) -> CsVecI<N, I>
    where F: FnMut(&N) -> N,
          N: Clone
    {
        let mut res = self.to_owned();
        res.map_inplace(f);
        res
    }
}

/// # Methods on sparse vectors with mutable access to their data
impl<'a, N, I, IStorage, DStorage> CsVecBase<IStorage, DStorage>
where N: 'a,
      I: 'a + SpIndex,
      IStorage: 'a + Deref<Target=[I]>,
      DStorage: DerefMut<Target=[N]> {

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
        }
        else {
            None
        }
    }

    /// Apply a function to each non-zero element, mutating it
    pub fn map_inplace<F>(&mut self, mut f: F)
    where F: FnMut(&N) -> N
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

}

/// # Methods propagating the lifetime of a `CsVecViewI`.
impl<'a, N: 'a, I: 'a + SpIndex> CsVecBase<&'a [I], &'a [N]> {

    /// Create a borrowed CsVec over slice data.
    pub fn new_view(
        n: usize,
        indices: &'a [I],
        data: &'a [N])
    -> Result<CsVecViewI<'a, N, I>, SprsError> {
        let v = CsVecViewI {
            dim: n,
            indices: indices,
            data: data,
        };
        v.check_structure().and(Ok(v))
    }

    /// Access element at given index, with logarithmic complexity
    ///
    /// Re-borrowing version of `at()`.
    pub fn get_rbr(&self, index: usize) -> Option<&'a N> {
        self.nnz_index(index).map(|NnzIndex(position)| {
            &self.data[position]
        })
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
    pub unsafe fn new_view_raw(n: usize,
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
where N: 'a,
      I: 'a + SpIndex
{

    /// Create a borrowed CsVec over slice data without checking the structure
    /// This is unsafe because algorithms are free to assume
    /// that properties guaranteed by check_structure are enforced, and
    /// because the lifetime of the pointers is unconstrained.
    /// For instance, non out-of-bounds indices can be relied upon to
    /// perform unchecked slice access.
    /// For safety, lifetime of the resulting vector should match the lifetime
    /// of the input pointers.
    pub unsafe fn new_view_mut_raw(n: usize,
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

impl<'a, 'b, N, I, IS1, DS1, IpS2, IS2, DS2>
Mul<&'b CsMatBase<N, I, IpS2, IS2, DS2>>
for &'a CsVecBase<IS1, DS1>
where N: 'a + Copy + Num + Default,
      I: 'a + SpIndex,
      IS1: 'a + Deref<Target=[I]>,
      DS1: 'a + Deref<Target=[N]>,
      IpS2: 'b + Deref<Target=[I]>,
      IS2: 'b + Deref<Target=[I]>,
      DS2: 'b + Deref<Target=[N]> {

    type Output = CsVecI<N, I>;

    fn mul(self, rhs: &CsMatBase<N, I, IpS2, IS2, DS2>) -> CsVecI<N, I> {
        (&self.row_view() * rhs).outer_view(0).unwrap().to_owned()
    }
}

impl<'a, 'b, N, I, IpS1, IS1, DS1, IS2, DS2>
Mul<&'b CsVecBase<IS2, DS2>>
for &'a CsMatBase<N, I, IpS1, IS1, DS1>
where N: Copy + Num + Default,
      I: SpIndex,
      IpS1: Deref<Target=[I]>,
      IS1: Deref<Target=[I]>,
      DS1: Deref<Target=[N]>,
      IS2: Deref<Target=[I]>,
      DS2: Deref<Target=[N]> {

    type Output = CsVecI<N, I>;

    fn mul(self, rhs: &CsVecBase<IS2, DS2>) -> CsVecI<N, I> {
        if self.is_csr() {
            prod::csr_mul_csvec(self.view(), rhs.view())
        }
        else {
            (self * &rhs.col_view()).outer_view(0).unwrap().to_owned()
        }
    }
}

impl<'a, 'b, N, IS1, DS1, IS2, DS2> Add<&'b CsVecBase<IS2, DS2>>
for &'a CsVecBase<IS1, DS1>
where N: Copy + Num,
      IS1: Deref<Target=[usize]>,
      DS1: Deref<Target=[N]>,
      IS2: Deref<Target=[usize]>,
      DS2: Deref<Target=[N]> {

    type Output = CsVec<N>;

    fn add(self, rhs: &CsVecBase<IS2, DS2>) -> CsVec<N> {
        binop::csvec_binop(self.view(),
                           rhs.view(),
                           |&x, &y| x + y
                          ).unwrap()
    }
}

impl<'a, 'b, N, IS1, DS1, IS2, DS2> Sub<&'b CsVecBase<IS2, DS2>>
for &'a CsVecBase<IS1, DS1>
where N: Copy + Num,
      IS1: Deref<Target=[usize]>,
      DS1: Deref<Target=[N]>,
      IS2: Deref<Target=[usize]>,
      DS2: Deref<Target=[N]> {

    type Output = CsVec<N>;

    fn sub(self, rhs: &CsVecBase<IS2, DS2>) -> CsVec<N> {
        binop::csvec_binop(self.view(),
                           rhs.view(),
                           |&x, &y| x - y
                          ).unwrap()
    }
}

impl<N, IS, DS> Index<usize> for CsVecBase<IS, DS>
where IS: Deref<Target=[usize]>,
      DS: Deref<Target=[N]> {

    type Output = N;

    fn index(&self, index: usize) -> &N {
        self.get(index).unwrap()
    }
}

impl<N, IS, DS> IndexMut<usize> for CsVecBase<IS, DS>
where IS: Deref<Target=[usize]>,
      DS: DerefMut<Target=[N]> {

    fn index_mut(&mut self, index: usize) -> &mut N {
        self.get_mut(index).unwrap()
    }
}

impl<N, IS, DS> Index<NnzIndex> for CsVecBase<IS, DS>
where IS: Deref<Target=[usize]>,
      DS: Deref<Target=[N]>
{
    type Output = N;

    fn index(&self, index: NnzIndex) -> &N {
        let NnzIndex(i) = index;
        self.data().get(i).unwrap()
    }
}

impl<N, IS, DS> IndexMut<NnzIndex> for CsVecBase<IS, DS>
where IS: Deref<Target=[usize]>,
      DS: DerefMut<Target=[N]>
{
    fn index_mut(&mut self, index: NnzIndex) -> &mut N {
        let NnzIndex(i) = index;
        self.data_mut().get_mut(i).unwrap()
    }
}

#[cfg(test)]
mod test {
    use sparse::{CsVec, CsVecI};
    use super::SparseIterTools;
    use ndarray::Array;

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
        let slice = &dense_vec[..];
        assert_eq!(16., vec1.dot(&dense_vec));
        assert_eq!(16., vec1.dot(slice));

        let ndarray_vec = Array::linspace(1., 8., 8);
        assert_eq!(16., vec1.dot(&ndarray_vec));
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

        let expected = CsVec::new(8,
                                  vec![0, 2, 4, 6],
                                  vec![1., 1., 2., 1.],);

        assert_eq!(vec, expected);

        vec[6] = 3.;

        let expected = CsVec::new(8,
                                  vec![0, 2, 4, 6],
                                  vec![1., 1., 2., 3.],);

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
        let mut vec = CsVec::new(8,
                                 vec![0, 2, 4, 6],
                                 vec![1., 2., 3., 4.]);
        vec.map_inplace(|&x| x + 1.);
        let expected = CsVec::new(8,
                                  vec![0, 2, 4, 6],
                                  vec![2., 3., 4., 5.]);
        assert_eq!(vec, expected);
    }

    #[test]
    fn map() {
        let vec = CsVec::new(8, vec![0, 2, 4, 6], vec![1., 2., 3., 4.]);
        let res = vec.map(|&x| x * 2.);
        let expected = CsVec::new(8,
                                  vec![0, 2, 4, 6],
                                  vec![2., 4., 6., 8.]);
        assert_eq!(res, expected);
    }

    #[test]
    fn iter_mut() {
        let mut vec = CsVec::new(8,
                                 vec![0, 2, 4, 6],
                                 vec![1., 2., 3., 4.]);
        for (ind, val) in vec.iter_mut() {
            if ind == 2 {
                *val += 1.;
            }
            else {
                *val *= 2.;
            }
        }
        let expected = CsVec::new(8,
                                  vec![0, 2, 4, 6],
                                  vec![2., 3., 6., 8.]);
        assert_eq!(vec, expected);
    }
}
