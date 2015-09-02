/// A sparse vector, which can be extracted from a sparse matrix
///

use std::iter::{Zip, Peekable, FilterMap};
use std::ops::{Deref, Mul};
use std::cmp;
use std::slice::{Iter};

use num::traits::Num;

use sparse::permutation::Permutation;
use sparse::prod;
use sparse::csmat::{CsMat, CsMatView};
use sparse::csmat::CompressedStorage::{CSR, CSC};

#[derive(PartialEq, Debug)]
pub struct CsVec<N, IStorage, DStorage>
where N: Clone,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {
    dim: usize,
    // FIXME: maybe CsMat could be more generic over its storage types
    // to avoid having to allocate extra fields to CsVec only to be able to
    // convert...
    indptr: [usize; 2],
    indices : IStorage,
    data : DStorage
}

pub type CsVecView<'a, N> = CsVec<N, &'a [usize], &'a [N]>;
pub type CsVecOwned<N> = CsVec<N, Vec<usize>, Vec<N>>;

pub struct VectorIterator<'a, N: 'a> {
    dim: usize,
    ind_data: Zip<Iter<'a,usize>, Iter<'a,N>>,
}

pub struct VectorIteratorPerm<'a, N: 'a> {
    dim: usize,
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


impl<'a, N: 'a + Copy> VectorIterator<'a, N> {


    /// Iterate over the matching non-zero elements of both vectors
    /// Useful for vector dot product
    pub fn nnz_zip<M>(self,
                     other: VectorIterator<'a, M>
                     )
     -> FilterMap<NnzOrZip<'a, N, M>, fn(NnzEither<N,M>) -> Option<(usize,N,M)>>
    where M: 'a + Copy {
        assert!(self.dim == other.dim);
        let nnz_or_iter = NnzOrZip {
            left: self.peekable(),
            right: other.peekable(),
        };
        nnz_or_iter.filter_map(filter_both_nnz)
    }

    pub fn nnz_or_zip<M>(self,
                         other: VectorIterator<'a, M>) -> NnzOrZip<'a, N, M>
    where M: 'a + Copy {
        assert!(self.dim == other.dim);
        NnzOrZip {
            left: self.peekable(),
            right: other.peekable(),
        }
    }
}


/// An iterator over the non zeros of either of two vector iterators, ordered, 
/// such that the sum of the vectors may be computed
pub struct NnzOrZip<'a, N1: 'a + Copy, N2: 'a + Copy> {
    left: Peekable<VectorIterator<'a, N1>>,
    right: Peekable<VectorIterator<'a, N2>>
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

impl <'a, N1: 'a + Copy, N2: 'a + Copy>
Iterator
for NnzOrZip<'a, N1, N2> {
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

impl<'a, N: 'a + Clone> CsVec<N, &'a[usize], &'a[N]> {

    pub fn new_borrowed(
        n: usize,
        indices: &'a [usize],
        data: &'a [N])
    -> CsVec<N, &'a[usize], &'a[N]> {
        CsVec {
            dim: n,
            indptr: [0, indices.len()],
            indices: indices,
            data: data,
        }
    }
}

impl<N: Clone> CsVec<N, Vec<usize>, Vec<N>> {
    pub fn new_owned(n: usize,
                     indices: Vec<usize>,
                     data: Vec<N>
                    ) -> CsVec<N, Vec<usize>, Vec<N>> {
        // FIXME: should check its structure
        CsVec {
            dim: n,
            indptr: [0, indices.len()],
            indices: indices,
            data: data
        }
    }

    pub fn empty(dim: usize) -> CsVec<N, Vec<usize>, Vec<N>> {
        CsVec {
            dim: dim,
            indptr: [0, 0],
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
            Some(&last_ind) => assert!(ind > last_ind)
        }
        assert!(ind <= self.dim);
        self.indptr[1] += 1;
        self.indices.push(ind);
        self.data.push(val);
    }

    pub fn reserve(&mut self, size: usize) {
        self.indices.reserve(size);
        self.data.reserve(size);
    }

    pub fn reserve_exact(&mut self, exact_size: usize) {
        self.indices.reserve_exact(exact_size);
        self.data.reserve_exact(exact_size);
    }

    pub fn clear(&mut self) {
        self.indices.clear();
        self.data.clear();
    }
}

impl<N, IStorage, DStorage> CsVec<N, IStorage, DStorage>
where N:  Copy,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {

    pub fn borrowed(&self) -> CsVec<N, &[usize], &[N]> {
        CsVec {
            dim: self.dim,
            indptr: self.indptr,
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }
}

impl<'a, N, IStorage, DStorage> CsVec<N, IStorage, DStorage>
where N: 'a + Copy,
IStorage: 'a + Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {

    pub fn iter(&self) -> VectorIterator<N> {
        VectorIterator {
            dim: self.dim,
            ind_data: self.indices.iter().zip(self.data.iter()),
        }
    }

    pub fn iter_perm<'perm: 'a>(&'a self,
                                perm: &'perm Permutation<&'perm [usize]>)
                               -> VectorIteratorPerm<'a, N> {
        VectorIteratorPerm {
            dim: self.dim,
            ind_data: self.indices.iter().zip(self.data.iter()),
            perm: perm.borrowed()
        }
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn data(&self) -> &[N] {
        &self.data
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn check_structure(&self) -> bool {
        self.indices.windows(2).all(|x| x[0] < x[1])
    }

    pub fn to_owned(&self) -> CsVecOwned<N> {
        CsVec {
            dim: self.dim,
            indptr: self.indptr,
            indices: self.indices.to_vec(),
            data: self.data.to_vec(),
        }
    }

    pub fn row_view(&self) -> CsMatView<N> {
        // TODO: don't check the structure (requires a structure check at
        // vec creation)
        CsMatView::from_slices(CSR, 1, self.dim,
                               &self.indptr[..],
                               &self.indices[..],
                               &self.data[..]).unwrap()
    }

    pub fn col_view(&self) -> CsMatView<N> {
        // TODO: don't check the structure (requires a structure check at
        // vec creation)
        CsMatView::from_slices(CSC, self.dim, 1,
                               &self.indptr[..],
                               &self.indices[..],
                               &self.data[..]).unwrap()
    }

    pub fn dot<IS2, DS2>(&self, rhs: &CsVec<N, IS2, DS2>) -> N
    where N: Num, IS2: Deref<Target=[usize]>, DS2: Deref<Target=[N]> {
        self.iter().nnz_zip(rhs.iter()).map(|(_, lval, rval)| lval * rval)
                                       .fold(N::zero(), |x, y| x + y)
    }
}

impl<'a, 'b, N, IS1, DS1, IS2, DS2> Mul<&'b CsMat<N, IS2, DS2>>
for &'a CsVec<N, IS1, DS1>
where N: 'a + Copy + Num + Default,
      IS1: 'a + Deref<Target=[usize]>,
      DS1: 'a + Deref<Target=[N]>,
      IS2: 'b + Deref<Target=[usize]>,
      DS2: 'b + Deref<Target=[N]> {

    type Output = CsVecOwned<N>;

    fn mul(self, rhs: &CsMat<N, IS2, DS2>) -> CsVecOwned<N> {
        (&self.row_view() * rhs).outer_view(0).unwrap().to_owned()
    }
}

impl<'a, 'b, N, IS1, DS1, IS2, DS2> Mul<&'b CsVec<N, IS2, DS2>>
for &'a CsMat<N, IS1, DS1>
where N: Copy + Num + Default,
      IS1: Deref<Target=[usize]>,
      DS1: Deref<Target=[N]>,
      IS2: Deref<Target=[usize]>,
      DS2: Deref<Target=[N]> {

    type Output = CsVecOwned<N>;

    fn mul(self, rhs: &CsVec<N, IS2, DS2>) -> CsVecOwned<N> {
        if self.is_csr() {
            prod::csr_mul_csvec(self.borrowed(), rhs.borrowed())
        }
        else {
            (self * &rhs.col_view()).outer_view(0).unwrap().to_owned()
        }
    }
}


#[cfg(test)]
mod test {
    use super::CsVec;

    fn test_vec1() -> CsVec<f64, Vec<usize>, Vec<f64>> {
        let n = 8;
        let indices = vec![0, 1, 4, 5, 7];
        let data = vec![0., 1., 4., 5., 7.];

        return CsVec::new_owned(n, indices, data);
    }

    fn test_vec2() -> CsVec<f64, Vec<usize>, Vec<f64>> {
        let n = 8;
        let indices = vec![0, 2, 4, 6, 7];
        let data = vec![0.5, 2.5, 4.5, 6.5, 7.5];

        return CsVec::new_owned(n, indices, data);
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
        let vec1 = CsVec::new_owned(8, vec![0, 2, 4, 6], vec![1.; 4]);
        let vec2 = CsVec::new_owned(8, vec![1, 3, 5, 7], vec![2.; 4]);
        let vec3 = CsVec::new_owned(8, vec![1, 2, 5, 6], vec![3.; 4]);

        assert_eq!(0., vec1.dot(&vec2));
        assert_eq!(4., vec1.dot(&vec1));
        assert_eq!(16., vec2.dot(&vec2));
        assert_eq!(6., vec1.dot(&vec3));
        assert_eq!(12., vec2.dot(&vec3));
    }
}
