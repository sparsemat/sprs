///! A sparse matrix in the Compressed Sparse Row/Column format
///
/// In the CSR format, a matrix is a structure containing three vectors:
/// indptr, indices, and data
/// These vectors satisfy the relation
/// for i in [0, nrows],
/// A(i, indices[indptr[i]..indptr[i+1]]) = data[indptr[i]..indptr[i+1]]
/// In the CSC format, the relation is
/// A(indices[indptr[i]..indptr[i+1]], i) = data[indptr[i]..indptr[i+1]]

use std::iter::{Enumerate};
use std::default::Default;
use std::slice::{Windows};
use std::ops::{Deref, DerefMut, Add, Sub, Mul};
use std::mem;
use num::traits::Num;

use sparse::permutation::{Permutation};
use sparse::vec::{CsVec, CsVecView};
use sparse::compressed::SpMatView;
use sparse::binop;
use sparse::prod;

pub type CsMatVec<N> = CsMat<N, Vec<usize>, Vec<N>>;
pub type CsMatView<'a, N> = CsMat<N, &'a [usize], &'a [N]>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompressedStorage {
    CSR,
    CSC
}

impl CompressedStorage {
    pub fn other_storage(&self) -> CompressedStorage {
        match *self {
            CSR => CSC,
            CSC => CSR,
        }
    }
}

use self::CompressedStorage::*;

/// Iterator on the matrix' outer dimension
/// Implemented over an iterator on the indptr array
pub struct OuterIterator<'iter, N: 'iter> {
    inner_len: usize,
    indptr_iter: Enumerate<Windows<'iter, usize>>,
    indices: &'iter [usize],
    data: &'iter [N],
}

/// Iterator on the matrix' outer dimension, permuted
/// Implemented over an iterator on the indptr array
pub struct OuterIteratorPerm<'iter, 'perm: 'iter, N: 'iter> {
    inner_len: usize,
    indptr_iter: Enumerate<Windows<'iter, usize>>,
    indices: &'iter [usize],
    data: &'iter [N],
    perm: Permutation<&'perm[usize]>,
}


/// Outer iteration on a compressed matrix yields
/// a tuple consisting of the outer index and of a sparse vector
/// containing the associated inner dimension
impl <'iter, N: 'iter + Clone>
Iterator
for OuterIterator<'iter, N> {
    type Item = (usize, CsVec<N, &'iter[usize], &'iter[N]>);
    #[inline]
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.indptr_iter.next() {
            None => None,
            Some((outer_ind, window)) => {
                let inner_start = window[0];
                let inner_end = window[1];
                let indices = &self.indices[inner_start..inner_end];
                let data = &self.data[inner_start..inner_end];
                let vec = CsVec::new_borrowed(
                    self.inner_len, indices, data);
                Some((outer_ind, vec))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indptr_iter.size_hint()
    }
}

/// Permuted outer iteration on a compressed matrix yields
/// a tuple consisting of the outer index and of a sparse vector
/// containing the associated inner dimension
impl <'iter, 'perm: 'iter, N: 'iter + Clone>
Iterator
for OuterIteratorPerm<'iter, 'perm, N> {
    type Item = (usize, CsVec<N, &'iter[usize], &'iter[N]>);
    #[inline]
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.indptr_iter.next() {
            None => None,
            Some((outer_ind, window)) => {
                let inner_start = window[0];
                let inner_end = window[1];
                let outer_ind_perm = self.perm.at(outer_ind);
                let indices = &self.indices[inner_start..inner_end];
                let data = &self.data[inner_start..inner_end];
                let vec = CsVec::new_borrowed(
                    self.inner_len, indices, data);
                Some((outer_ind_perm, vec))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indptr_iter.size_hint()
    }
}

/// Reverse outer iteration on a compressed matrix yields
/// a tuple consisting of the outer index and of a sparse vector
/// containing the associated inner dimension
///
/// Only the outer dimension iteration is reverted. If you wish to also
/// revert the inner dimension, you should call rev() again when iterating
/// the vector.
impl <'iter, N: 'iter + Clone>
DoubleEndedIterator
for OuterIterator<'iter, N> {
    #[inline]
    fn next_back(&mut self) -> Option<<Self as Iterator>::Item> {
        match self.indptr_iter.next_back() {
            None => None,
            Some((outer_ind, window)) => {
                let inner_start = window[0];
                let inner_end = window[1];
                let indices = &self.indices[inner_start..inner_end];
                let data = &self.data[inner_start..inner_end];
                let vec = CsVec::new_borrowed(self.inner_len, indices, data);
                Some((outer_ind, vec))
            }
        }
    }
}

impl <'iter, N: 'iter + Clone> ExactSizeIterator for OuterIterator<'iter, N> {
    fn len(&self) -> usize {
        self.indptr_iter.len()
    }
}

#[derive(PartialEq, Debug)]
pub struct CsMat<N, IndStorage, DataStorage>
where IndStorage: Deref<Target=[usize]>, DataStorage: Deref<Target=[N]> {
    storage: CompressedStorage,
    nrows : usize,
    ncols : usize,
    nnz : usize,
    indptr : IndStorage,
    indices : IndStorage,
    data : DataStorage
}

impl<'a, N:'a + Copy> CsMat<N, &'a[usize], &'a[N]> {
    /// Create a borrowed CsMat matrix from sliced data,
    /// checking their validity
    pub fn from_slices(
        storage: CompressedStorage, nrows : usize, ncols: usize,
        indptr : &'a[usize], indices : &'a[usize], data : &'a[N]
        )
    -> Option<CsMat<N, &'a[usize], &'a[N]>> {
        let m = CsMat {
            storage: storage,
            nrows : nrows,
            ncols: ncols,
            nnz : data.len(),
            indptr : indptr,
            indices : indices,
            data : data,
        };
        match m.check_compressed_structure() {
            None => None,
            _ => Some(m)
        }
    }
}

impl<N: Copy> CsMat<N, Vec<usize>, Vec<N>> {
    /// Create an empty CsMat for building purposes
    pub fn empty(storage: CompressedStorage, inner_size: usize
                ) -> CsMat<N, Vec<usize>, Vec<N>> {
        let (nrows, ncols) = match storage {
            CSR => (0, inner_size),
            CSC => (inner_size, 0)
        };
        CsMat {
            storage: storage,
            nrows: nrows,
            ncols: ncols,
            nnz: 0,
            indptr: vec![0; 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn zero(rows: usize, cols: usize) -> CsMatVec<N> {
        CsMat {
            storage: CSR,
            nrows: rows,
            ncols: cols,
            nnz: 0,
            indptr: vec![0; rows + 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Reserve the storage for the given additional number of nonzero data
    pub fn reserve_outer_dim(&mut self, outer_dim_additional: usize) {
        self.indptr.reserve(outer_dim_additional);
    }

    /// Reserve the storage for the given additional number of nonzero data
    pub fn reserve_nnz(&mut self, nnz_additional: usize) {
        self.indices.reserve(nnz_additional);
        self.data.reserve(nnz_additional);
    }

    /// Reserve the storage for the given number of nonzero data
    pub fn reserve_outer_dim_exact(&mut self, outer_dim_lim: usize) {
        self.indptr.reserve_exact(outer_dim_lim + 1);
    }

    /// Reserve the storage for the given number of nonzero data
    pub fn reserve_nnz_exact(&mut self, nnz_lim: usize) {
        self.indices.reserve_exact(nnz_lim);
        self.data.reserve_exact(nnz_lim);
    }

    /// Create an owned CsMat matrix from moved data,
    /// checking their validity
    pub fn from_vecs(
        storage: CompressedStorage, nrows : usize, ncols: usize,
        indptr : Vec<usize>, indices : Vec<usize>, data : Vec<N>
        )
    -> Option<CsMat<N, Vec<usize>, Vec<N>>> {
        let m = CsMat {
            storage: storage,
            nrows : nrows,
            ncols: ncols,
            nnz : data.len(),
            indptr : indptr,
            indices : indices,
            data : data,
        };
        match m.check_compressed_structure() {
            None => None,
            _ => Some(m)
        }
    }

    /// Append an outer dim to an existing matrix, compressing it in the process
    pub fn append_outer(mut self, data: &[N]) -> Self where N: Num {
        for (inner_ind, val) in data.iter().enumerate() {
            if *val != N::zero() {
                self.indices.push(inner_ind);
                self.data.push(*val);
                self.nnz += 1;
            }
        }
        match self.storage {
            CSR => self.nrows += 1,
            CSC => self.ncols += 1
        }
        self.indptr.push(self.nnz);
        self
    }

    /// Append an outer dim to an existing matrix, provided by a sparse vector
    pub fn append_outer_csvec(mut self, vec: CsVec<N,&[usize],&[N]>) -> Self {
        assert_eq!(self.inner_dims(), vec.dim());
        for (ind, val) in vec.indices().iter().zip(vec.data()) {
            self.indices.push(*ind);
            self.data.push(*val);
        }
        match self.storage {
            CSR => self.nrows += 1,
            CSC => self.ncols += 1
        }
        self.nnz += vec.nnz();
        self.indptr.push(self.nnz);
        self
    }
}

impl<N: Num + Copy> CsMat<N, Vec<usize>, Vec<N>> {
    /// Identity matrix
    pub fn eye(storage: CompressedStorage, dim: usize
              ) -> CsMat<N, Vec<usize>, Vec<N>> {
        let n = dim;
        let indptr = (0..n+1).collect();
        let indices = (0..n).collect();
        let data = vec![N::one(); n];
        CsMat {
            storage: storage,
            nrows: n,
            ncols: n,
            nnz: n,
            indptr: indptr,
            indices: indices,
            data: data,
        }
    }

}

impl<N, IndStorage, DataStorage> CsMat<N, IndStorage, DataStorage>
where N: Copy,
      IndStorage: Deref<Target=[usize]>,
      DataStorage: Deref<Target=[N]> {

    /// Return an outer iterator for the matrix
    pub fn outer_iterator<'a>(&'a self) -> OuterIterator<'a, N> {
        let inner_len = match self.storage {
            CSR => self.ncols,
            CSC => self.nrows
        };
        OuterIterator {
            inner_len: inner_len,
            indptr_iter: self.indptr.windows(2).enumerate(),
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }

    /// Return an outer iterator over P*A, as well as the proper permutation
    /// for iterating over the inner dimension of P*A*P^T
    pub fn outer_iterator_perm<'a, 'perm: 'a>(
        &'a self, perm: &'perm Permutation<&'perm [usize]>)
    -> OuterIteratorPerm<'a, 'perm, N> {
        let (inner_len, oriented_perm) = match self.storage {
            CSR => (self.ncols, perm.borrowed()),
            CSC => (self.nrows, Permutation::inv(perm))
        };
        OuterIteratorPerm {
            inner_len: inner_len,
            indptr_iter: self.indptr.windows(2).enumerate(),
            indices: &self.indices[..],
            data: &self.data[..],
            perm: oriented_perm
        }
    }

    pub fn storage(&self) -> CompressedStorage {
        self.storage
    }

    pub fn rows(&self) -> usize {
        self.nrows
    }

    pub fn cols(&self) -> usize {
        self.ncols
    }

    pub fn nb_nonzero(&self) -> usize {
        self.nnz
    }

    pub fn outer_dims(&self) -> usize {
        match self.storage {
            CSR => self.nrows,
            CSC => self.ncols
        }
    }

    pub fn inner_dims(&self) -> usize {
        match self.storage {
            CSC => self.nrows,
            CSR => self.ncols
        }
    }

    pub fn at(&self, &(i,j) : &(usize, usize)) -> Option<N> {
        assert!(i < self.nrows);
        assert!(j < self.ncols);

        match self.storage {
            CSR => self.at_outer_inner(&(i,j)),
            CSC => self.at_outer_inner(&(j,i))
        }
    }

    /// Get a view into the i-th outer dimension (eg i-th row for a CSR matrix)
    pub fn outer_view(&self, i: usize) -> Option<CsVecView<N>> {
        if i >= self.outer_dims() {
            return None;
        }
        let start = self.indptr[i];
        let stop = self.indptr[i+1];
        Some(CsVecView::new_borrowed(self.inner_dims(),
                                     &self.indices[start..stop],
                                     &self.data[start..stop]))
    }

    pub fn indptr(&self) -> &[usize] {
        &self.indptr[..]
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices[..]
    }

    pub fn data(&self) -> &[N] {
        &self.data[..]
    }

    pub fn is_csc(&self) -> bool {
        self.storage == CSC
    }

    pub fn is_csr(&self) -> bool {
        self.storage == CSR
    }

    /// Transpose a matrix in place
    /// No allocation required (this is simply a storage order change)
    pub fn transpose_mut(&mut self) {
        mem::swap(&mut self.nrows, &mut self.ncols);
        self.storage = self.storage.other_storage();
    }

    /// Transpose a matrix in place
    /// No allocation required (this is simply a storage order change)
    pub fn transpose_into(mut self) -> Self {
        self.transpose_mut();
        self
    }

    /// Transposed view of this matrix
    /// No allocation required (this is simply a storage order change)
    pub fn transpose_view(&self) -> CsMatView<N> {
        CsMatView {
            storage: self.storage.other_storage(),
            nrows: self.ncols,
            ncols: self.nrows,
            nnz: self.nnz,
            indptr: &self.indptr[..],
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }

    pub fn to_owned(&self) -> CsMatVec<N> {
        CsMatVec {
            storage: self.storage,
            nrows: self.nrows,
            ncols: self.ncols,
            nnz: self.nnz,
            indptr: self.indptr.to_vec(),
            indices: self.indices.to_vec(),
            data: self.data.to_vec(),
        }
    }

    pub fn at_outer_inner(&self, &(outer_ind, inner_ind): &(usize, usize))
    -> Option<N> {
        let begin = self.indptr[outer_ind];
        let end = self.indptr[outer_ind+1];
        if begin >= end {
            return None;
        }
        let indices = &self.indices[begin..end];
        let data = &self.data[begin..end];

        let position = match indices.binary_search(&inner_ind) {
            Ok(ind) => ind,
            _ => return None
        };

        Some(data[position].clone())
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
        if self.indices.iter().max().unwrap_or(&0) >= &inner {
            println!("CsMat indices values incoherent with ncols");
            return None;
        }

        if ! self.indptr.deref().windows(2).all(|x| x[0] <= x[1]) {
            println!("CsMat indptr not sorted");
            return None;
        }

        // check that the indices are sorted for each row
        if ! self.outer_iterator().all(
            | (_, vec) | { vec.check_structure() })
        {
            println!("CsMat indices not sorted for each outer ind");
            return None;
        }

        Some(nnz)
    }

    /// Return a view into the current matrix
    pub fn borrowed(&self) -> CsMat<N, &[usize], &[N]> {
        CsMat {
            storage: self.storage,
            nrows: self.nrows,
            ncols: self.ncols,
            nnz: self.nnz,
            indptr: &self.indptr[..],
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }
}

impl<N, IndStorage, DataStorage> CsMat<N, IndStorage, DataStorage>
where N: Copy + Default,
      IndStorage: Deref<Target=[usize]>,
      DataStorage: Deref<Target=[N]> {

    /// Create a matrix mathematically equal to this one, but with the
    /// opposed storage.
    pub fn to_other_storage(&self) -> CsMat<N, Vec<usize>, Vec<N>> {
        let mut indptr = vec![0; self.inner_dims() + 1];
        let mut indices = vec![0; self.nb_nonzero()];
        let mut data = vec![N::default(); self.nb_nonzero()];
        let borrowed = self.borrowed();
        raw::convert_mat_storage(borrowed,
                                 &mut indptr, &mut indices, &mut data);
        CsMat::from_vecs(self.storage().other_storage(),
                         self.rows(), self.cols(),
                         indptr, indices, data).unwrap()
    }

    pub fn to_csc(&self) -> CsMatVec<N> {
        match self.storage {
            CSR => self.to_other_storage(),
            CSC => self.to_owned()
        }
    }

    pub fn to_csr(&self) -> CsMatVec<N> {
        match self.storage {
            CSR => self.to_owned(),
            CSC => self.to_other_storage()
        }
    }

}

impl<N, IndStorage, DataStorage> CsMat<N, IndStorage, DataStorage>
where
N: Copy,
IndStorage: DerefMut<Target=[usize]>,
DataStorage: DerefMut<Target=[N]> {

    pub fn data_mut(&mut self) -> &mut [N] {
        &mut self.data[..]
    }

    /// Sparse matrix self-multiplication by a scalar
    pub fn scale(&mut self, val: N) where N: Num + Copy {
        for data in self.data_mut() {
            *data = *data * val;
        }
    }

}

mod raw {
    use super::CsMat;
    use std::mem::swap;

    /// Copy-convert a CsMat into the oppposite storage.
    /// Can be used to implement CSC <-> CSR conversions, or to implement
    /// same-storage (copy) transposition.
    ///
    /// # Panics
    /// Panics if indptr contains non-zero values
    ///
    /// Panics if the output slices don't match the input matrices'
    /// corresponding slices.
    pub fn convert_mat_storage<N: Copy>(mat: CsMat<N, &[usize], &[N]>,
                                        indptr: &mut [usize],
                                        indices: &mut[usize],
                                        data: &mut [N]) {
        assert_eq!(indptr.len(), mat.inner_dims() + 1);
        assert_eq!(indices.len(), mat.indices().len());
        assert_eq!(data.len(), mat.data().len());

        assert!(indptr.iter().all(|x| *x == 0));

        for (_, vec) in mat.outer_iterator() {
            for (inner_dim, _) in vec.iter() {
                indptr[inner_dim] += 1;
            }
        }

        let mut cumsum = 0;
        for iptr in indptr.iter_mut() {
            let tmp = *iptr;
            *iptr = cumsum;
            cumsum += tmp;
        }
        if let Some(last_iptr) = indptr.last() {
            assert_eq!(*last_iptr, mat.nb_nonzero());
        }

        for (outer_dim, vec) in mat.outer_iterator() {
            for (inner_dim, val) in vec.iter() {
                let dest = indptr[inner_dim];
                data[dest] = val;
                indices[dest] = outer_dim;
                indptr[inner_dim] += 1;
            }
        }

        let mut last = 0;
        for iptr in indptr.iter_mut() {
            swap(iptr, &mut last);
        }
    }
}

impl<'a, 'b, N, IStorage, DStorage, Mat> Add<&'b Mat>
for &'a CsMat<N, IStorage, DStorage>
where N: 'a + Copy + Num + Default,
      IStorage: 'a + Deref<Target=[usize]>,
      DStorage: 'a + Deref<Target=[N]>,
      Mat: SpMatView<N> {
    type Output = CsMatVec<N>;

    fn add(self, rhs: &'b Mat) -> CsMatVec<N> {
        if self.storage() != rhs.borrowed().storage() {
            return binop::add_mat_same_storage(
                self, &rhs.borrowed().to_other_storage()).unwrap()
        }
        binop::add_mat_same_storage(self, rhs).unwrap()
    }
}

impl<'a, 'b, N, IStorage, DStorage, Mat> Sub<&'b Mat>
for &'a CsMat<N, IStorage, DStorage>
where N: 'a + Copy + Num + Default,
      IStorage: 'a + Deref<Target=[usize]>,
      DStorage: 'a + Deref<Target=[N]>,
      Mat: SpMatView<N> {
    type Output = CsMatVec<N>;

    fn sub(self, rhs: &'b Mat) -> CsMatVec<N> {
        if self.storage() != rhs.borrowed().storage() {
            return binop::sub_mat_same_storage(
                self, &rhs.borrowed().to_other_storage()).unwrap()
        }
        binop::sub_mat_same_storage(self, rhs).unwrap()
    }
}

impl<'a,N, IStorage, DStorage> Mul<N>
for &'a CsMat<N, IStorage, DStorage>
where N: 'a + Copy + Num,
      IStorage: 'a + Deref<Target=[usize]>,
      DStorage: 'a + Deref<Target=[N]> {
    type Output = CsMatVec<N>;

    fn mul(self, rhs: N) -> CsMatVec<N> {
        binop::scalar_mul_mat(self, rhs)
    }
}

impl<'a, 'b, N, IS1, DS1, IS2, DS2> Mul<&'b CsMat<N, IS2, DS2>>
for &'a CsMat<N, IS1, DS1>
where N: 'a + Copy + Num + Default,
      IS1: 'a + Deref<Target=[usize]>,
      DS1: 'a + Deref<Target=[N]>,
      IS2: 'b + Deref<Target=[usize]>,
      DS2: 'b + Deref<Target=[N]> {
    type Output = CsMatVec<N>;

    fn mul(self, rhs: &'b CsMat<N, IS2, DS2>) -> CsMatVec<N> {
        match (self.storage(), rhs.storage()) {
            (CSR, CSR) => {
                let mut workspace = prod::workspace_csr(self, rhs);
                prod::csr_mul_csr(self, rhs, &mut workspace).unwrap()
            }
            (CSR, CSC) => {
                let mut workspace = prod::workspace_csr(self, rhs);
                prod::csr_mul_csr(self,
                                  &rhs.to_other_storage(),
                                  &mut workspace).unwrap()
            }
            (CSC, CSR) => {
                let mut workspace = prod::workspace_csc(self, rhs);
                prod::csc_mul_csc(&self.to_other_storage(), rhs,
                                  &mut workspace).unwrap()
            }
            (CSC, CSC) => {
                let mut workspace = prod::workspace_csc(self, rhs);
                prod::csc_mul_csc(self, rhs, &mut workspace).unwrap()
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::{CsMat};
    use super::CompressedStorage::{CSC, CSR};
    use test_data::{mat1, mat1_csc, mat1_times_2};

    #[test]
    fn test_new_csr_success() {
        let indptr_ok : &[usize] = &[0, 1, 2, 3];
        let indices_ok : &[usize] = &[0, 1, 2];
        let data_ok : &[f64] = &[1., 1., 1.];
        match CsMat::from_slices(CSR, 3, 3, indptr_ok, indices_ok, data_ok) {
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
        match CsMat::from_slices(CSR, 3, 3, indptr_fail1, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match CsMat::from_slices(CSR, 3, 3, indptr_fail2, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match CsMat::from_slices(CSR, 3, 3, indptr_fail3, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match CsMat::from_slices(CSR, 3, 3, indptr_ok, indices_fail1, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match CsMat::from_slices(CSR, 3, 3, indptr_ok, indices_fail2, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match CsMat::from_slices(CSR, 3, 3, indptr_ok, indices_ok, data_fail1) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match CsMat::from_slices(CSR, 3, 3, indptr_ok, indices_ok, data_fail2) {
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
        match CsMat::from_slices(CSR, 5, 5, indptr, indices, data) {
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
        match CsMat::from_slices(CSR, 3, 4, indptr_ok, indices_ok, data_ok) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
        match CsMat::from_slices(CSC, 4, 3, indptr_ok, indices_ok, data_ok) {
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
        match CsMat::from_slices(CSR, 4, 3, indptr_ok, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match CsMat::from_slices(CSC, 3, 4, indptr_ok, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
    }


    #[test]
    fn test_new_csr_vec_borrowed() {
        let indptr_ok = vec![0, 1, 2, 3];
        let indices_ok = vec![0, 1, 2];
        let data_ok : Vec<f64> = vec![1., 1., 1.];
        match CsMat::from_slices(CSR, 3, 3, &indptr_ok, &indices_ok, &data_ok) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
    }

    #[test]
    fn test_new_csr_vec_owned() {
        let indptr_ok = vec![0, 1, 2, 3];
        let indices_ok = vec![0, 1, 2];
        let data_ok : Vec<f64> = vec![1., 1., 1.];
        match CsMat::from_vecs(CSR, 3, 3, indptr_ok, indices_ok, data_ok) {
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
        match CsMat::from_slices(CSR, 5, 5, indptr, indices, data) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
    }

    #[test]
    fn csr_to_csc() {
        let a = mat1();
        let a_csc_ground_truth = mat1_csc();
        let a_csc = a.to_other_storage();
        assert_eq!(a_csc, a_csc_ground_truth);
    }

    #[test]
    fn test_self_smul() {
        let mut a = mat1();
        a.scale(2.);
        let c_true = mat1_times_2();
        assert_eq!(a.indptr(), c_true.indptr());
        assert_eq!(a.indices(), c_true.indices());
        assert_eq!(a.data(), c_true.data());
    }
}
