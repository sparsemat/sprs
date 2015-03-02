/// A sparse vector, which can be extracted from a sparse matrix
///

use std::iter::{Zip};
use std::ops::{Deref};
use std::slice::{Iter, SliceExt};

use sparse::permutation::Permutation;

pub struct CsVec<'perm, N, IStorage, DStorage>
where N: 'perm + Clone,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {
    indices : IStorage,
    data : DStorage,
    perm: Permutation<&'perm [usize]>,
}

pub struct VectorIterator<'perm, N: 'perm> {
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


impl<'perm, N: 'perm + Clone> CsVec<'perm, N, &'perm[usize], &'perm[N]> {

    pub fn new_borrowed(
        indices: &'perm [usize],
        data: &'perm [N],
        perm: Permutation<&'perm [usize]>)
    -> CsVec<'perm, N, &'perm[usize], &'perm[N]> {
        CsVec {
            indices: indices,
            data: data,
            perm: perm,
        }
    }
}

impl<'perm, N, IStorage, DStorage> CsVec<'perm, N, IStorage, DStorage>
where N: 'perm + Clone,
IStorage: Deref<Target=[usize]>,
DStorage: Deref<Target=[N]> {

    pub fn iter(&self) -> VectorIterator<N> {
        VectorIterator {
            ind_data: self.indices.iter().zip(self.data.iter()),
            perm: self.perm.borrowed()
        }
    }

    pub fn indices(&self) -> &[usize] {
        self.indices.as_slice()
    }

    pub fn data(&self) -> &[N] {
        self.data.as_slice()
    }

    pub fn check_structure(&self) -> bool {
        self.indices.windows(2).all(|x| x[0] < x[1])
    }
}
