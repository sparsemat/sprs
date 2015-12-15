///! Compressed matrices with unsorted inner indices
///!
///! These matrices are mostly an implementation detail of some algorithms
///! (LU factorization, triplet format to csr/csc, ...) but can be useful
///! in their own right to interface with third party code.

use sparse::csmat::{self, CompressedStorage};

/// A view of an unsorted compressed matrix
///
/// TODO: it could be a good idea to define CsMatView as a newtype
/// over this, this would enable a working Deref
pub struct CMatView<'a, N: 'a> {
    storage: CompressedStorage,
    nrows : usize,
    ncols : usize,
    nnz : usize,
    indptr : &'a [usize],
    indices : &'a [usize],
    data : &'a [N]
}

impl<'a, N> CMatView<'a, N> {

    pub fn new(storage: CompressedStorage,
               nrows: usize,
               ncols: usize,
               indptr: &'a [usize],
               indices: &'a [usize],
               data: &'a [N])
               -> CMatView<'a, N> {
        let outer_dims = csmat::outer_dimension(storage, nrows, ncols);
        let nnz = indptr[outer_dims];
        assert!(nnz <= indices.len());
        assert!(nnz <= data.len());
        CMatView {
            storage: storage,
            nrows: nrows,
            ncols: ncols,
            nnz: nnz,
            indptr: indptr,
            indices: indices,
            data: data
        }
   }

}
