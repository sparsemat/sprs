///! A sparse matrix in the Compressed Sparse Row/Column format
///
/// In the CSR format, a matrix is a structure containing three vectors:
/// indptr, indices, and data
/// These vectors satisfy the relation
/// for i in [0, nrows],
/// A(i, indices[indptr[i]..indptr[i+1]]) = data[indptr[i]..indptr[i+1]]
/// In the CSC format, the relation is
/// A(indices[indptr[i]..indptr[i+1]], i) = data[indptr[i]..indptr[i+1]]

use std::iter::{Peekable};
use std::slice::{Iter, SliceExt};
use std::borrow::{Cow, IntoCow};
use std::ops::{Deref};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CompressedStorage {
    CSR,
    CSC
}

/// Iterator on the matrix' outer dimension
/// Implemented over an iterator on the indptr array
pub struct OuterIterator<'a, N: 'a> {
    outer_ind: usize,
    indptr_iter: Peekable<&'a usize, Iter<'a, usize>>,
    indices: &'a [usize],
    data: &'a [N]
}

/// Outer iteration on a compressed matrix yields
/// a tuple consisting of the outer index and the corresponding
/// inner indices and associated data
impl <'a, N: 'a>
Iterator
for OuterIterator<'a, N> {
    type Item = (usize, &'a[usize], &'a[N]);
    #[inline]
    fn next(&mut self) -> Option<(usize, &'a[usize], &'a[N])> {
        let cur_index = match self.indptr_iter.next() {
            None => { return None; },
            Some(value) => value
        };
        match self.indptr_iter.peek() {
            None => None,
            Some(next_index) => {
                self.outer_ind += 1;
                // FIXME double dereference is ugly...
                // this seems to show something is wrong with the types
                return Some((self.outer_ind - 1,
                        &self.indices[*cur_index..**next_index],
                        &self.data[*cur_index..**next_index]));
            }
        }
    }
}

pub struct CsMat<'a, N: 'a + Clone> {
    storage: CompressedStorage,
    nrows : usize,
    ncols : usize,
    nnz : usize,
    indptr : Cow<'a, Vec<usize>, [usize]>,
    indices : Cow<'a, Vec<usize>, [usize]>,
    data : Cow<'a, Vec<N>, [N]>,
}

/// Create a CsMat matrix from its main components, checking their validity
/// Validity check is performed using check_compressed_structure()
pub fn new_borrowed_csmat<'a, N: Clone>(
        storage: CompressedStorage, nrows : usize, ncols: usize,
        indptr : &'a[usize], indices : &'a[usize], data : &'a[N]
        ) -> Option<CsMat<'a, N>> {
    let m = CsMat {
        storage: storage,
        nrows : nrows,
        ncols: ncols,
        nnz : data.len(),
        indptr : indptr.into_cow(),
        indices : indices.into_cow(),
        data : data.into_cow(),
    };
    match m.check_compressed_structure() {
        None => None,
        _ => Some(m)
    }
}

pub fn new_csmat<'a, N: 'a + Clone>(
        storage: CompressedStorage, nrows : usize, ncols: usize,
        indptr : Vec<usize>, indices : Vec<usize>, data : Vec<N>
        ) -> Option<CsMat<'a, N>> {
    let m = CsMat {
        storage: storage,
        nrows : nrows,
        ncols: ncols,
        nnz : data.len(),
        indptr : indptr.into_cow(),
        indices : indices.into_cow(),
        data : data.into_cow(),
    };
    match m.check_compressed_structure() {
        None => None,
        _ => Some(m)
    }
}

impl<'a, N: 'a + Clone> CsMat<'a, N> {

    /// Return an outer iterator for the matrix
    pub fn outer_iterator(&'a self) -> OuterIterator<'a, N> {
        OuterIterator {
            outer_ind: 0us,
            indptr_iter: self.indptr.iter().peekable(),
            indices: self.indices.as_slice(),
            data: self.data.as_slice()
        }
    }

    pub fn storage_type(&self) -> CompressedStorage {
        self.storage
    }

    pub fn rows(&self) -> usize {
        self.nrows
    }

    pub fn cols(&self) -> usize {
        self.ncols
    }

    /// Check the structure of CsMat components
    fn check_compressed_structure(&self) -> Option<usize> {
        let inner = match self.storage {
            CompressedStorage::CSR => self.ncols,
            CompressedStorage::CSC => self.nrows
        };
        let outer = match self.storage {
            CompressedStorage::CSR => self.nrows,
            CompressedStorage::CSC => self.ncols
        };
        if self.indptr.len() != outer + 1 {
            println!("CsMat indptr length incorrect");
            return None;
        }
        if self.indices.len() != self.data.len() {
            println!("CsMat indices/data length incorrect");
            return None;
        }
        let nnz = self.indices.len();
        if nnz != self.nnz {
            println!("CsMat nnz count incorrect");
            return None;
        }
        if self.indptr.iter().max().unwrap() > &nnz {
            println!("CsMat indptr values incoherent with nnz");
            return None;
        }
        if self.indices.iter().max().unwrap() >= &inner {
            println!("CsMat indices values incoherent with ncols");
            return None;
        }

        if ! self.indptr.deref().windows(2).all(|&: x| x[0] <= x[1]) {
            println!("CsMat indptr not sorted");
            return None;
        }

        // check that the indices are sorted for each row
        if ! self.outer_iterator().all(
            | &: (_, inds, _) | {
                inds.windows(2).all(|&: x| x[0] < x[1])
            }) {
            println!("CsMat indices not sorted for each outer ind");
            return None;
        }

        Some(nnz)
    }
}


#[cfg(test)]
mod test {
    use super::{new_borrowed_csmat, new_csmat};
    use super::CompressedStorage::{CSC, CSR};

    #[test]
    fn test_new_csr_success() {
        let indptr_ok : &[usize] = &[0, 1, 2, 3];
        let indices_ok : &[usize] = &[0, 1, 2];
        let data_ok : &[f64] = &[1., 1., 1.];
        match new_borrowed_csmat(CSR, 3, 3, indptr_ok, indices_ok, data_ok) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
    }

    #[test]
    fn test_new_csr_fails() {
        let indptr_ok : &[usize] = &[0, 1, 2, 3];
        let indices_ok : &[usize] = &[0, 1, 2];
        let data_ok : &[f64] = &[1., 1., 1.];
        let indptr_fail1 : &[usize] = &[0, 1, 2];
        let indptr_fail2 : &[usize] = &[0, 1, 2, 4];
        let indptr_fail3 : &[usize] = &[0, 2, 1, 3];
        let indices_fail1 : &[usize] = &[0, 1];
        let indices_fail2 : &[usize] = &[0, 1, 4];
        let data_fail1 : &[f64] = &[1., 1., 1., 1.];
        let data_fail2 : &[f64] = &[1., 1.,];
        match new_borrowed_csmat(CSR, 3, 3, indptr_fail1, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csmat(CSR, 3, 3, indptr_fail2, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csmat(CSR, 3, 3, indptr_fail3, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csmat(CSR, 3, 3, indptr_ok, indices_fail1, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csmat(CSR, 3, 3, indptr_ok, indices_fail2, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csmat(CSR, 3, 3, indptr_ok, indices_ok, data_fail1) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csmat(CSR, 3, 3, indptr_ok, indices_ok, data_fail2) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
    }

    #[test]
    fn test_new_csr_fail_indices_ordering() {
        let indptr: &[usize] = &[0, 2, 4, 5, 6, 7];
        // good indices would be [2, 3, 3, 4, 2, 1, 3];
        let indices: &[usize] = &[3, 2, 3, 4, 2, 1, 3];
        let data: &[f64] = &[
            0.35310881, 0.42380633, 0.28035896, 0.58082095,
            0.53350123, 0.88132896, 0.72527863];
        match new_borrowed_csmat(CSR, 5, 5, indptr, indices, data) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
    }

    #[test]
    fn test_new_csr_csc_success() {
        let indptr_ok : &[usize] = &[0, 2, 5, 6];
        let indices_ok : &[usize] = &[2, 3, 1, 2, 3, 3];
        let data_ok : &[f64] = &[
            0.05734571, 0.15543348, 0.75628258,
            0.83054515, 0.71851547, 0.46202352];
        match new_borrowed_csmat(CSR, 3, 4, indptr_ok, indices_ok, data_ok) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
        match new_borrowed_csmat(CSC, 4, 3, indptr_ok, indices_ok, data_ok) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
    }

    #[test]
    fn test_new_csr_csc_fails() {
        let indptr_ok : &[usize] = &[0, 2, 5, 6];
        let indices_ok : &[usize] = &[2, 3, 1, 2, 3, 3];
        let data_ok : &[f64] = &[
            0.05734571, 0.15543348, 0.75628258,
            0.83054515, 0.71851547, 0.46202352];
        match new_borrowed_csmat(CSR, 4, 3, indptr_ok, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csmat(CSC, 3, 4, indptr_ok, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
    }


    #[test]
    fn test_new_csr_vec_borrowed() {
        let indptr_ok = vec![0us, 1, 2, 3];
        let indices_ok = vec![0us, 1, 2];
        let data_ok : Vec<f64> = vec![1., 1., 1.];
        match new_borrowed_csmat(CSR, 3, 3, indptr_ok.as_slice(),
                      indices_ok.as_slice(), data_ok.as_slice()) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
    }

    #[test]
    fn test_new_csr_vec_owned() {
        let indptr_ok = vec![0us, 1, 2, 3];
        let indices_ok = vec![0us, 1, 2];
        let data_ok : Vec<f64> = vec![1., 1., 1.];
        match new_csmat(CSR, 3, 3, indptr_ok, indices_ok, data_ok) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
    }

    #[test]
    fn new_csr_with_empty_row() {
        let indptr: &[usize] = &[0, 3, 3, 5, 6, 7];
        let indices: &[usize] = &[1, 2, 3, 2, 3, 4, 4];
        let data: &[f64] = &[
            0.75672424, 0.1649078, 0.30140296, 0.10358244,
            0.6283315, 0.39244208, 0.57202407];
        match new_borrowed_csmat(CSR, 5, 5, indptr, indices, data) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
    }
}
