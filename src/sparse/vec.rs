/// A sparse vector, which can be extracted from a sparse matrix
///

use std::iter::{Zip};
use std::ops::{Deref};
use std::cmp;
use std::slice::{Iter};

use sparse::permutation::Permutation;

pub struct CsVec<N, IStorage, DStorage>
where N: Clone,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {
    len: usize,
    indices : IStorage,
    data : DStorage,
    perm: Permutation<IStorage>,
}

pub struct VectorIterator<'perm, N: 'perm> {
    len: usize,
    ind_data: Zip<Iter<'perm,usize>, Iter<'perm,N>>,
    perm: Permutation<&'perm [usize]>,
}


impl <'perm, N: 'perm + Copy>
Iterator
for VectorIterator<'perm, N> {
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

impl<'perm, N: 'perm + Copy> VectorIterator<'perm, N> {
    pub fn nnz_zip<M>(self,
                     other: VectorIterator<'perm, M>
                     ) -> NnzZip<'perm, N, M>
    where M: 'perm + Clone {
        assert!(self.len == other.len);
        NnzZip {
            left: self,
            right: other
        }
    }
}


/// An iterator the iterates over the matching non-zeros of two
/// vector iterators, hence enabling eg computing their dot-product
pub struct NnzZip<'perm, N1: 'perm, N2: 'perm> {
    left: VectorIterator<'perm, N1>,
    right: VectorIterator<'perm, N2>
}

impl <'perm, N1: 'perm + Copy, N2: 'perm + Copy>
Iterator
for NnzZip<'perm, N1, N2> {
    type Item = (usize, N1, N2);

    fn next(&mut self) -> Option<(usize, N1, N2)> {
        match (self.left.next(), self.right.next()) {
            (None, _) => None,
            (_, None) => None,
            (Some((mut lind, mut lval)), Some((mut rind, mut rval))) => {
                loop {
                    if lind < rind {
                        let lnext = self.left.next();
                        if lnext.is_none() {
                            return None;
                        }
                        let (lind_, lval_) = lnext.unwrap();
                        lind = lind_;
                        lval = lval_;
                    }
                    else if rind < lind {
                        let rnext = self.right.next();
                        if rnext.is_none() {
                            return None;
                        }
                        let (rind_, rval_) = rnext.unwrap();
                        rind = rind_;
                        rval = rval_;
                    }
                    else {
                        println!("{} {}", lind, rind);
                        return Some((lind, lval, rval));
                    }
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, left_upper) = self.left.size_hint();
        let (_, right_upper) = self.right.size_hint();
        let upper = match (left_upper, right_upper) {
            (Some(x), Some(y)) => Some(cmp::min(x,y)),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None
        };
        (0, upper)
    }
}


impl<'perm, N: 'perm + Clone> CsVec<N, &'perm[usize], &'perm[N]> {

    pub fn new_borrowed(
        n: usize,
        indices: &'perm [usize],
        data: &'perm [N],
        perm: Permutation<&'perm [usize]>)
    -> CsVec<N, &'perm[usize], &'perm[N]> {
        CsVec {
            len: n,
            indices: indices,
            data: data,
            perm: perm,
        }
    }
}

impl<N: Clone> CsVec<N, Vec<usize>, Vec<N>> {
    pub fn new_owned(n: usize,
                     indices: Vec<usize>,
                     data: Vec<N>,
                     perm: Permutation<Vec<usize>>
                    ) -> CsVec<N, Vec<usize>, Vec<N>> {
        CsVec {
            len: n,
            indices: indices,
            data: data,
            perm: perm
        }
    }
}

impl<N, IStorage, DStorage> CsVec<N, IStorage, DStorage>
where N:  Clone,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {

    fn borrowed(&self) -> CsVec<N, &[usize], &[N]> {
        CsVec {
            len: self.len,
            indices: &self.indices[..],
            data: &self.data[..],
            perm: self.perm.borrowed()
        }
    }
}

impl<'perm, N, IStorage, DStorage> CsVec<N, IStorage, DStorage>
where N: 'perm + Clone,
IStorage: 'perm + Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {

    pub fn iter(&self) -> VectorIterator<N> {
        VectorIterator {
            len: self.len,
            ind_data: self.indices.iter().zip(self.data.iter()),
            perm: self.perm.borrowed()
        }
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn data(&self) -> &[N] {
        &self.data
    }

    pub fn check_structure(&self) -> bool {
        self.indices.windows(2).all(|x| x[0] < x[1])
    }
}

#[cfg(test)]
mod test {
    use super::CsVec;
    use super::NnzZip;
    use sparse::permutation::Permutation;

    fn test_vec1() -> CsVec<f64, Vec<usize>, Vec<f64>> {
        let n = 8;
        let indices = vec![0, 1, 4, 5, 7];
        let data = vec![0., 1., 4., 5., 7.];

        return CsVec::new_owned(n, indices, data, Permutation::identity());
    }

    fn test_vec2() -> CsVec<f64, Vec<usize>, Vec<f64>> {
        let n = 8;
        let indices = vec![0, 2, 4, 6, 7];
        let data = vec![0.5, 1.5, 4.5, 6.5, 7.5];

        return CsVec::new_owned(n, indices, data, Permutation::identity());
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
}
