use std::ops::Deref;
use indexing::SpIndex;

pub use self::csmat::{CompressedStorage};

/// Compressed matrix in the CSR or CSC format, with sorted indices.
///
/// This sparse matrix format is the preferred format for performing arithmetic
/// operations. Constructing a sparse matrix directly in this format requires
/// a deep knowledge of its internals. For easier matrix construction, the
/// [triplet format](struct.TripletMatBase) is preferred.
///
/// The `CsMatBase` type is parameterized by the scalar type `N`, the indexing
/// type `I`, the indexing storage backend types `IptrStorage` and `IndStorage`,
/// and the value storage backend type `DataStorage`. Convenient aliases are
/// available to specify frequent variants: [`CsMat`] refers to a sparse matrix
/// that owns its data, similar to `Vec<T>`; [`CsMatView`] refers to a sparse matrix
/// that borrows its data, similar to `& [T]`; and [`CsMatViewMut`] refers to a sparse
/// matrix borrowing its data, with a mutable borrow for its values. No mutable
/// borrow is allowed for the structure of the matrix, allowing the invariants
/// to be preserved.
///
/// Additionaly, the type aliases [`CsMatI`], [`CsMatViewI`] and
/// [`CsMatViewMutI`] can be used to choose an index type different from the
/// default `usize`.
///
/// [`CsMat`]: type.CsMat.html
/// [`CsMatView`]: type.CsMatView.html
/// [`CsMatViewMut`]: type.CsMatViewMut.html
/// [`CsMatI`]: type.CsMatI.html
/// [`CsMatViewI`]: type.CsMatViewI.html
/// [`CsMatViewMutI`]: type.CsMatViewMutI.html
///
/// ## Storage format
///
/// In the compressed storage format, the non-zero values of a sparse matrix
/// are stored as the row and column location of the non-zero values, with
/// a compression along the rows (CSR) or columns (CSC) indices. The dimension
/// along which the storage is compressed is referred to as the *outer dimension*,
/// the other dimension is called the *inner dimension*. For clarity, the
/// remaining explanation will assume a CSR matrix, but the information stands
/// for CSC matrices as well.
///
/// ### Indptr
///
/// An index pointer array `indptr` of size corresponding to the number of rows
/// stores the cumulative sum of non-zero elements for each row. For instance,
/// the number of non-zero elements of the i-th row can be obtained by computing
/// `indptr[i + 1] - indptr[i]`. The total number of non-zero elements is thus
/// `nnz = indptr[nb_rows + 1]`. This index pointer array can then be used to
/// efficiently index the `indices` and `data` array, which respectively contain
/// the column indices and the values of the non-zero elements.
///
/// ### Indices and data
///
/// The non-zero locations and values are stored in arrays of size `nnz`, `indices`
/// and `data`. For row `i`, the non-zeros are located in the slices
/// `indices[indptr[i]..indptr[i+1]]` and `data[indptr[i]..indptr[i+1]]`. We
/// require and enforce sorted indices for each row.
///
/// ## Construction
///
/// A sparse matrix can be directly constructed by providing its index pointer,
/// indices and data arrays.
#[derive(PartialEq, Debug)]
pub struct CsMatBase<N, I, IptrStorage, IndStorage, DataStorage>
where I: SpIndex,
      IptrStorage: Deref<Target=[I]>,
      IndStorage: Deref<Target=[I]>,
      DataStorage: Deref<Target=[N]> {
    storage: CompressedStorage,
    nrows : usize,
    ncols : usize,
    indptr : IptrStorage,
    indices : IndStorage,
    data : DataStorage
}

pub type CsMatI<N, I> = CsMatBase<N, I, Vec<I>, Vec<I>, Vec<N>>;
pub type CsMatViewI<'a, N, I> = CsMatBase<N, I, &'a [I], &'a [I], &'a [N]>;
pub type CsMatViewMutI<'a, N, I> = CsMatBase<N, I, &'a [I], &'a [I], &'a mut [N]>;
pub type CsMatVecView_<'a, N, I> = CsMatBase<N, I, Vec<I>, &'a [I], &'a [N]>;

pub type CsMat<N> = CsMatI<N, usize>;
pub type CsMatView<'a, N> = CsMatViewI<'a, N, usize>;
pub type CsMatViewMut<'a, N> = CsMatViewMutI<'a, N, usize>;
// FIXME: a fixed size array would be better, but no Deref impl
pub type CsMatVecView<'a, N> = CsMatVecView_<'a, N, usize>;

/// A sparse vector, storing the indices of its non-zero data.
/// The indices should be sorted.
#[derive(PartialEq, Debug)]
pub struct CsVecBase<N, IStorage, DStorage>
where DStorage: Deref<Target=[N]> {
    dim: usize,
    indices : IStorage,
    data : DStorage
}

pub type CsVecViewI<'a, N, I> = CsVecBase<N, &'a [I], &'a [N]>;
pub type CsVecViewMut_<'a, N, I> = CsVecBase<N, &'a [I], &'a mut [N]>;
pub type CsVecI<N, I> = CsVecBase<N, Vec<I>, Vec<N>>;

pub type CsVecView<'a, N> = CsVecViewI<'a, N, usize>;
pub type CsVecViewMut<'a, N> = CsVecViewMut_<'a, N, usize>;
pub type CsVec<N> = CsVecI<N, usize>;

mod prelude {
    pub use super::{
        CsMatBase,
        CsMatViewI,
        CsMatView,
        CsMatViewMutI,
        CsMatViewMut,
        CsMatI,
        CsMat,
        CsMatVecView_,
        CsMatVecView,
        CsVecBase,
        CsVecViewI,
        CsVecView,
        CsVecViewMut_,
        CsVecViewMut,
        CsVecI,
        CsVec,
    };
}

mod utils {
    use indexing::SpIndex;

    pub fn sort_indices_data_slices<N: Copy, I:SpIndex>(indices: &mut [I],
                                                        data: &mut [N],
                                                        buf: &mut Vec<(I, N)>) {
        let len = indices.len();
        assert_eq!(len, data.len());
        let indices = &mut indices[..len];
        let data = &mut data[..len];
        buf.clear();
        buf.reserve_exact(len);
        for i in 0..len {
            buf.push((indices[i], data[i]));
        }

        buf.sort_by_key(|x| x.0);

        for (i, &(ind, x)) in buf.iter().enumerate() {
            indices[i] = ind;
            data[i] = x;
        }
    }
}

pub mod csmat;
pub mod triplet;
pub mod vec;
pub mod permutation;
pub mod prod;
pub mod binop;
pub mod construct;
pub mod linalg;
pub mod symmetric;
pub mod compressed;
pub mod to_dense;
