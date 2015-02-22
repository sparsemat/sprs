/// A sparse vector, which can be extracted from a sparse matrix
///

use std::iter::{Zip};
use std::ops::{Deref};
use std::slice::{Iter, SliceExt};

use sparse::permutation::Permutation;

pub struct CsVec<'a, N, IStorage, DStorage>
where N: 'a + Clone,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {
    indices : IStorage,
    data : DStorage,
    perm: &'a Permutation,
}

pub struct VectorIterator<'a, N: 'a> {
    ind_data: Zip<Iter<'a,usize>, Iter<'a,N>>,
    perm: &'a Permutation,
}


impl <'a, N: 'a + Copy>
Iterator
for VectorIterator<'a, N> {
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


impl<'a, N: 'a + Clone> CsVec<'a, N, &'a[usize], &'a[N]> {

    pub fn new_borrowed(
        indices: &'a [usize], data: &'a [N], perm: &'a Permutation)
    -> CsVec<'a, N, &'a[usize], &'a[N]> {
        CsVec {
            indices: indices,
            data: data,
            perm: perm,
        }
    }
}

impl<'a, N, IStorage, DStorage> CsVec<'a, N, IStorage, DStorage>
where N: 'a + Clone,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {

    pub fn iter(&self) -> VectorIterator<N> {
        VectorIterator {
            ind_data: self.indices.iter().zip(self.data.iter()),
            perm: &self.perm
        }
    }

    pub fn indices(&self) -> &[usize] {
        self.indices.as_slice()
    }

    pub fn data(&self) -> &[N] {
        self.data.as_slice()
    }

    pub fn check_structure(&self) -> bool {
        self.indices.windows(2).all(|&: x| x[0] < x[1])
    }
}
