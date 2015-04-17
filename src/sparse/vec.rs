/// A sparse vector, which can be extracted from a sparse matrix
///

use std::iter::{Zip};
use std::ops::{Deref};
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
