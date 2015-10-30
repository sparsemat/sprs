/// A sparse vector, which can be extracted from a sparse matrix
///
/// # Example
/// ```rust
/// use sprs::CsVec;
/// let vec1 = CsVec::new_owned(8, vec![0, 2, 5, 6], vec![1.; 4]).unwrap();
/// let vec2 = CsVec::new_owned(8, vec![1, 3, 5], vec![2.; 3]).unwrap();
/// let res = &vec1 + &vec2;
/// let mut iter = res.iter();
/// assert_eq!(iter.next(), Some((0, 1.)));
/// assert_eq!(iter.next(), Some((1, 2.)));
/// assert_eq!(iter.next(), Some((2, 1.)));
/// assert_eq!(iter.next(), Some((3, 2.)));
/// assert_eq!(iter.next(), Some((5, 3.)));
/// assert_eq!(iter.next(), Some((6, 1.)));
/// assert_eq!(iter.next(), None);
/// ```

use std::iter::{Zip, Peekable, FilterMap, IntoIterator, Enumerate};
use std::ops::{Deref, Mul, Add, Sub};
use std::cmp;
use std::slice::{self, Iter};
use std::collections::HashSet;
use std::hash::Hash;

use num::traits::Num;

use sparse::permutation::Permutation;
use sparse::{prod, binop};
use sparse::csmat::{CsMat, CsMatVecView};
use sparse::csmat::CompressedStorage::{CSR, CSC};
use errors::SprsError;

/// A sparse vector, storing the indices of its non-zero data.
/// The indices should be sorted.
#[derive(PartialEq, Debug)]
pub struct CsVec<N, IStorage, DStorage>
where DStorage: Deref<Target=[N]> {
    dim: usize,
    indices : IStorage,
    data : DStorage
}

pub type CsVecView<'a, N> = CsVec<N, &'a [usize], &'a [N]>;
pub type CsVecOwned<N> = CsVec<N, Vec<usize>, Vec<N>>;

/// A trait to represent types which can be interpreted as vectors
/// of a given dimension.
pub trait VecDim {
    /// The dimension of the vector
    fn dim(&self) -> usize;
}

impl<N, IS, DS: Deref<Target=[N]>> VecDim for CsVec<N, IS, DS> {
    fn dim(&self) -> usize {
        self.dim
    }
}

impl<N> VecDim for [N] {
    fn dim(&self) -> usize {
        self.len()
    }
}

impl<N> VecDim for Vec<N> {
    fn dim(&self) -> usize {
        self.len()
    }
}


/// An iterator over the non-zero elements of a sparse vector
pub struct VectorIterator<'a, N: 'a> {
    ind_data: Zip<Iter<'a,usize>, Iter<'a,N>>,
}

pub struct VectorIteratorPerm<'a, N: 'a> {
    ind_data: Zip<Iter<'a,usize>, Iter<'a,N>>,
    perm: Permutation<&'a [usize]>,
}


impl <'a, N: 'a + Copy>
Iterator
for VectorIterator<'a, N> {
    type Item = (usize, N);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.ind_data.next() {
            None => None,
            Some((inner_ind, data)) => Some((*inner_ind, *data))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ind_data.size_hint()
    }
}

impl <'a, N: 'a + Copy>
Iterator
for VectorIteratorPerm<'a, N> {
    type Item = (usize, N);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.ind_data.next() {
            None => None,
            Some((inner_ind, data)) => Some(
                (self.perm.at_inv(*inner_ind), *data))
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
    /// use sprs::sparse::vec::NnzEither;
    /// use sprs::sparse::vec::SparseIterTools;
    /// let v0 = CsVec::new_owned(5, vec![0, 2, 4], vec![1., 2., 3.]).unwrap();
    /// let v1 = CsVec::new_owned(5, vec![1, 2, 3], vec![-1., -2., -3.]
    ///                          ).unwrap();
    /// let mut nnz_or_iter = v0.iter().nnz_or_zip(v1.iter());
    /// assert_eq!(nnz_or_iter.next(), Some(NnzEither::Left((0, 1.))));
    /// assert_eq!(nnz_or_iter.next(), Some(NnzEither::Right((1, -1.))));
    /// assert_eq!(nnz_or_iter.next(), Some(NnzEither::Both((2, 2., -2.))));
    /// assert_eq!(nnz_or_iter.next(), Some(NnzEither::Right((3, -3.))));
    /// assert_eq!(nnz_or_iter.next(), Some(NnzEither::Left((4, 3.))));
    /// assert_eq!(nnz_or_iter.next(), None);
    /// ```
    fn nnz_or_zip<I, N1: Copy, N2: Copy>(self, other: I)
    -> NnzOrZip<Self, I::IntoIter, N1, N2>
    where Self: Iterator<Item=(usize, N1)> + Sized,
          I: IntoIterator<Item=(usize, N2)> {
        NnzOrZip {
            left: self.peekable(),
            right: other.into_iter().peekable(),
        }
    }

    /// Iterate over the matching non-zero elements of both vectors
    /// Useful for vector dot product.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sprs::CsVec;
    /// use sprs::sparse::vec::SparseIterTools;
    /// let v0 = CsVec::new_owned(5, vec![0, 2, 4], vec![1., 2., 3.]).unwrap();
    /// let v1 = CsVec::new_owned(5, vec![1, 2, 3], vec![-1., -2., -3.]
    ///                          ).unwrap();
    /// let mut nnz_zip = v0.iter().nnz_zip(v1.iter());
    /// assert_eq!(nnz_zip.next(), Some((2, 2., -2.)));
    /// assert_eq!(nnz_zip.next(), None);
    /// ```
    fn nnz_zip<I, N1: Copy, N2: Copy>(self, other: I)
    -> FilterMap<NnzOrZip<Self, I::IntoIter, N1, N2>,
                 fn(NnzEither<N1,N2>) -> Option<(usize,N1,N2)>>
    where Self: Iterator<Item=(usize, N1)> + Sized,
          I: IntoIterator<Item=(usize, N2)> {
        let nnz_or_iter = NnzOrZip {
            left: self.peekable(),
            right: other.into_iter().peekable(),
        };
        nnz_or_iter.filter_map(filter_both_nnz)
    }
}

impl<T: Iterator> SparseIterTools for Enumerate<T>
where <T as Iterator>::Item : Copy {
}

impl<'a, N: 'a + Copy> SparseIterTools for VectorIterator<'a, N> {
}

/// An iterator over the non zeros of either of two vector iterators, ordered,
/// such that the sum of the vectors may be computed
pub struct NnzOrZip<Ite1, Ite2, N1: Copy, N2: Copy>
where Ite1: Iterator<Item=(usize, N1)>,
      Ite2: Iterator<Item=(usize, N2)> {
    left: Peekable<Ite1>,
    right: Peekable<Ite2>,
}

#[derive(PartialEq, Debug)]
pub enum NnzEither<N1, N2> {
    Both((usize, N1, N2)),
    Left((usize, N1)),
    Right((usize, N2))
}

fn filter_both_nnz<N: Copy, M: Copy>(elem: NnzEither<N,M>)
-> Option<(usize, N, M)> {
    match elem {
        NnzEither::Both((ind, lval, rval)) => Some((ind, lval, rval)),
        _ => None
    }
}

impl <Ite1, Ite2, N1: Copy, N2: Copy>
Iterator
for NnzOrZip<Ite1, Ite2, N1, N2>
where Ite1: Iterator<Item=(usize, N1)>,
      Ite2: Iterator<Item=(usize, N2)> {
    type Item = NnzEither<N1, N2>;

    fn next(&mut self) -> Option<(NnzEither<N1, N2>)> {
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

impl<'a, N: 'a + Copy> CsVec<N, &'a[usize], &'a[N]> {

    /// Create a borrowed CsVec over slice data.
    pub fn new_borrowed(
        n: usize,
        indices: &'a [usize],
        data: &'a [N])
    -> Result<CsVec<N, &'a[usize], &'a[N]>, SprsError> {
        let v = CsVec {
            dim: n,
            indices: indices,
            data: data,
        };
        v.check_structure().and(Ok(v))
    }

    /// Create a borrowed CsVec over slice data without checking the structure
    /// This is unsafe because algorithms are free to assume
    /// that properties guaranteed by check_structure are enforced.
    /// For instance, non out-of-bounds indices can be relied upon to
    /// perform unchecked slice access.
    pub unsafe fn new_raw(n: usize,
                                         nnz: usize,
                                         indices: *const usize,
                                         data: *const N,
                                        ) -> CsVec<N, &'a[usize], &'a[N]> {
        CsVec {
            dim: n,
            indices: slice::from_raw_parts(indices, nnz),
            data: slice::from_raw_parts(data, nnz),
        }
    }
}

impl<N: Copy> CsVec<N, Vec<usize>, Vec<N>> {
    /// Create an owning CsVec from vector data.
    pub fn new_owned(n: usize,
                     indices: Vec<usize>,
                     data: Vec<N>
                    ) -> Result<CsVec<N, Vec<usize>, Vec<N>>, SprsError> {
        let v = CsVec {
            dim: n,
            indices: indices,
            data: data
        };
        v.check_structure().and(Ok(v))
    }

    /// Create an empty CsVec, which can be used for incremental construction
    pub fn empty(dim: usize) -> CsVec<N, Vec<usize>, Vec<N>> {
        CsVec {
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
    /// Panics if `ind` is lower or equal to the last
    /// element of `self.indices()`
    /// Panics if `ind` is greater than `self.dim()`
    pub fn append(&mut self, ind: usize, val: N) {
        match self.indices.last() {
            None => (),
            Some(&last_ind) => assert!(ind > last_ind, "unsorted append")
        }
        assert!(ind <= self.dim, "out of bounds index");
        self.indices.push(ind);
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

impl<N, IStorage, DStorage> CsVec<N, IStorage, DStorage>
where N:  Copy,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {

    /// Get a view of this vector.
    pub fn borrowed(&self) -> CsVecView<N> {
        CsVec {
            dim: self.dim,
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }
}

impl<'a, N, IStorage, DStorage> CsVec<N, IStorage, DStorage>
where N: 'a + Copy,
IStorage: 'a + Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {

    /// Iterate over the non zero values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sprs::CsVec;
    /// let v = CsVec::new_owned(5, vec![0, 2, 4], vec![1., 2., 3.]).unwrap();
    /// let mut iter = v.iter();
    /// assert_eq!(iter.next(), Some((0, 1.)));
    /// assert_eq!(iter.next(), Some((2, 2.)));
    /// assert_eq!(iter.next(), Some((4, 3.)));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> VectorIterator<N> {
        VectorIterator {
            ind_data: self.indices.iter().zip(self.data.iter()),
        }
    }

    /// Permuted iteration. Not finished
    #[doc(hidden)]
    pub fn iter_perm<'perm: 'a>(&'a self,
                                perm: &'perm Permutation<&'perm [usize]>)
                               -> VectorIteratorPerm<'a, N> {
        VectorIteratorPerm {
            ind_data: self.indices.iter().zip(self.data.iter()),
            perm: perm.borrowed()
        }
    }

    /// The underlying indices.
    pub fn indices(&self) -> &[usize] {
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

        if self.indices.iter().max().unwrap_or(&0) >= &self.dim {
            return Err(SprsError::OutOfBoundsIndex);
        }

        Ok(())
    }

    /// Allocate a new vector equal to this one.
    pub fn to_owned(&self) -> CsVecOwned<N> {
        CsVec {
            dim: self.dim,
            indices: self.indices.to_vec(),
            data: self.data.to_vec(),
        }
    }

    /// View this vector as a matrix with only one row.
    pub fn row_view(&self) -> CsMatVecView<N> {
        // Safe because we're taking a view into a vector that has
        // necessarily been checked
        let indptr = vec![0, self.indices.len()];
        unsafe {
            CsMatVecView::new_vecview_raw(CSR, 1, self.dim,
                                          indptr,
                                          self.indices.as_ptr(),
                                          self.data.as_ptr())
        }
    }

    /// View this vector as a matrix with only one column.
    pub fn col_view(&self) -> CsMatVecView<N> {
        // Safe because we're taking a view into a vector that has
        // necessarily been checked
        let indptr = vec![0, self.indices.len()];
        unsafe {
            CsMatVecView::new_vecview_raw(CSC, self.dim, 1,
                                          indptr,
                                          self.indices.as_ptr(),
                                          self.data.as_ptr())
        }
    }

    /// Access element at given index, with logarithmic complexity
    ///
    /// TODO: use this for CsMat::at_outer_inner
    pub fn at(&self, index: usize) -> Option<N> {
        let position = match self.indices.binary_search(&index) {
            Ok(ind) => ind,
            _ => return None
        };

        Some(self.data[position])
    }

    /// Vector dot product
    ///
    /// # Example
    ///
    /// ```rust
    /// use sprs::CsVec;
    /// let v1 = CsVec::new_owned(8, vec![1, 2, 4, 6], vec![1.; 4]).unwrap();
    /// let v2 = CsVec::new_owned(8, vec![1, 3, 5, 7], vec![2.; 4]).unwrap();
    /// assert_eq!(2., v1.dot(&v2));
    /// assert_eq!(4., v1.dot(&v1));
    /// assert_eq!(16., v2.dot(&v2));
    /// ```
    pub fn dot<IS2, DS2>(&self, rhs: &CsVec<N, IS2, DS2>) -> N
    where N: Num, IS2: Deref<Target=[usize]>, DS2: Deref<Target=[N]> {
        self.iter().nnz_zip(rhs.iter()).map(|(_, lval, rval)| lval * rval)
                                       .fold(N::zero(), |x, y| x + y)
    }

    /// Fill a dense vector with our values
    pub fn scatter(&self, out: &mut [N]) {
        for (ind, val) in self.iter() {
            out[ind] = val;
        }
    }

    /// Transform this vector into a set of (index, value) tuples
    pub fn to_set(self) -> HashSet<(usize, N)>
    where N: Hash + Eq {
        self.indices().iter().cloned().zip(self.data.iter().cloned()).collect()
    }
}

impl<'a, 'b, N, IS1, DS1, IpS2, IS2, DS2> Mul<&'b CsMat<N, IpS2, IS2, DS2>>
for &'a CsVec<N, IS1, DS1>
where N: 'a + Copy + Num + Default,
      IS1: 'a + Deref<Target=[usize]>,
      DS1: 'a + Deref<Target=[N]>,
      IpS2: 'b + Deref<Target=[usize]>,
      IS2: 'b + Deref<Target=[usize]>,
      DS2: 'b + Deref<Target=[N]> {

    type Output = CsVecOwned<N>;

    fn mul(self, rhs: &CsMat<N, IpS2, IS2, DS2>) -> CsVecOwned<N> {
        (&self.row_view() * rhs).outer_view(0).unwrap().to_owned()
    }
}

impl<'a, 'b, N, IpS1, IS1, DS1, IS2, DS2> Mul<&'b CsVec<N, IS2, DS2>>
for &'a CsMat<N, IpS1, IS1, DS1>
where N: Copy + Num + Default,
      IpS1: Deref<Target=[usize]>,
      IS1: Deref<Target=[usize]>,
      DS1: Deref<Target=[N]>,
      IS2: Deref<Target=[usize]>,
      DS2: Deref<Target=[N]> {

    type Output = CsVecOwned<N>;

    fn mul(self, rhs: &CsVec<N, IS2, DS2>) -> CsVecOwned<N> {
        if self.is_csr() {
            prod::csr_mul_csvec(self.borrowed(), rhs.borrowed()).unwrap()
        }
        else {
            (self * &rhs.col_view()).outer_view(0).unwrap().to_owned()
        }
    }
}

impl<'a, 'b, N, IS1, DS1, IS2, DS2> Add<&'b CsVec<N, IS2, DS2>>
for &'a CsVec<N, IS1, DS1>
where N: Copy + Num,
      IS1: Deref<Target=[usize]>,
      DS1: Deref<Target=[N]>,
      IS2: Deref<Target=[usize]>,
      DS2: Deref<Target=[N]> {

    type Output = CsVecOwned<N>;

    fn add(self, rhs: &CsVec<N, IS2, DS2>) -> CsVecOwned<N> {
        let binop = |x, y| x + y;
        binop::csvec_binop(self.borrowed(), rhs.borrowed(), binop).unwrap()
    }
}

impl<'a, 'b, N, IS1, DS1, IS2, DS2> Sub<&'b CsVec<N, IS2, DS2>>
for &'a CsVec<N, IS1, DS1>
where N: Copy + Num,
      IS1: Deref<Target=[usize]>,
      DS1: Deref<Target=[N]>,
      IS2: Deref<Target=[usize]>,
      DS2: Deref<Target=[N]> {

    type Output = CsVecOwned<N>;

    fn sub(self, rhs: &CsVec<N, IS2, DS2>) -> CsVecOwned<N> {
        let binop = |x, y| x - y;
        binop::csvec_binop(self.borrowed(), rhs.borrowed(), binop).unwrap()
    }
}


#[cfg(test)]
mod test {
    use super::CsVec;
    use super::SparseIterTools;

    fn test_vec1() -> CsVec<f64, Vec<usize>, Vec<f64>> {
        let n = 8;
        let indices = vec![0, 1, 4, 5, 7];
        let data = vec![0., 1., 4., 5., 7.];

        return CsVec::new_owned(n, indices, data).unwrap();
    }

    fn test_vec2() -> CsVec<f64, Vec<usize>, Vec<f64>> {
        let n = 8;
        let indices = vec![0, 2, 4, 6, 7];
        let data = vec![0.5, 2.5, 4.5, 6.5, 7.5];

        return CsVec::new_owned(n, indices, data).unwrap();
    }

    #[test]
    fn test_nnz_zip_iter() {
        let vec1 = test_vec1();
        let vec2 = test_vec2();
        let mut iter = vec1.iter().nnz_zip(vec2.iter());
        assert_eq!(iter.next().unwrap(), (0, 0., 0.5));
        assert_eq!(iter.next().unwrap(), (4, 4., 4.5));
        assert_eq!(iter.next().unwrap(), (7, 7., 7.5));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_nnz_or_zip_iter() {
        use super::NnzEither::*;
        let vec1 = test_vec1();
        let vec2 = test_vec2();
        let mut iter = vec1.iter().nnz_or_zip(vec2.iter());
        assert_eq!(iter.next().unwrap(), Both((0, 0., 0.5)));
        assert_eq!(iter.next().unwrap(), Left((1, 1.)));
        assert_eq!(iter.next().unwrap(), Right((2, 2.5)));
        assert_eq!(iter.next().unwrap(), Both((4, 4., 4.5)));
        assert_eq!(iter.next().unwrap(), Left((5, 5.)));
        assert_eq!(iter.next().unwrap(), Right((6, 6.5)));
        assert_eq!(iter.next().unwrap(), Both((7, 7., 7.5)));
    }

    #[test]
    fn dot_product() {
        let vec1 = CsVec::new_owned(8, vec![0, 2, 4, 6], vec![1.; 4]).unwrap();
        let vec2 = CsVec::new_owned(8, vec![1, 3, 5, 7], vec![2.; 4]).unwrap();
        let vec3 = CsVec::new_owned(8, vec![1, 2, 5, 6], vec![3.; 4]).unwrap();

        assert_eq!(0., vec1.dot(&vec2));
        assert_eq!(4., vec1.dot(&vec1));
        assert_eq!(16., vec2.dot(&vec2));
        assert_eq!(6., vec1.dot(&vec3));
        assert_eq!(12., vec2.dot(&vec3));
    }
}
