use array_backend::Array2;
use indexing::SpIndex;
use std::ops::Deref;

pub use self::csmat::CompressedStorage;

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
/// indices and data arrays. The coherence of the provided structure is then
/// verified.
///
/// For situations where the compressed structure is hard to figure out up front,
/// the [triplet format](struct.TriMatBase.html) can be used. A matrix in the
/// triplet format can then be efficiently converted to a `CsMat`.
///
/// Alternately, a sparse matrix can be constructed from other sparse matrices
/// using [`vstack`], [`hstack`] or [`bmat`].
///
/// [`vstack`]: fn.vstack.html
/// [`hstack`]: fn.hstack.html
/// [`bmat`]: fn.bmat.html
#[derive(Eq, PartialEq, Debug, Clone)]
#[cfg_attr(
    all(feature = "serde", feature = "serde_derive"),
    derive(Serialize, Deserialize)
)]
pub struct CsMatBase<N, I, IptrStorage, IndStorage, DataStorage, Iptr = I>
where
    I: SpIndex,
    Iptr: SpIndex,
    IptrStorage: Deref<Target = [Iptr]>,
    IndStorage: Deref<Target = [I]>,
    DataStorage: Deref<Target = [N]>,
{
    storage: CompressedStorage,
    nrows: usize,
    ncols: usize,
    indptr: IptrStorage,
    indices: IndStorage,
    data: DataStorage,
}

pub type CsMatI<N, I, Iptr = I> =
    CsMatBase<N, I, Vec<Iptr>, Vec<I>, Vec<N>, Iptr>;
pub type CsMatViewI<'a, N, I, Iptr = I> =
    CsMatBase<N, I, &'a [Iptr], &'a [I], &'a [N], Iptr>;
pub type CsMatViewMutI<'a, N, I, Iptr = I> =
    CsMatBase<N, I, &'a [Iptr], &'a [I], &'a mut [N], Iptr>;
pub type CsMatVecView_<'a, N, I, Iptr = I> =
    CsMatBase<N, I, Array2<Iptr>, &'a [I], &'a [N], Iptr>;

pub type CsMat<N> = CsMatI<N, usize>;
pub type CsMatView<'a, N> = CsMatViewI<'a, N, usize>;
pub type CsMatViewMut<'a, N> = CsMatViewMutI<'a, N, usize>;
// FIXME: a fixed size array would be better, but no Deref impl
pub type CsMatVecView<'a, N> = CsMatVecView_<'a, N, usize>;

/// A sparse vector, storing the indices of its non-zero data.
///
/// A `CsVec` represents a sparse vector by storing a sorted `indices()` array
/// containing the locations of the non-zero values and a `data()` array
/// containing the corresponding values. The format is compatible with `CsMat`,
/// ie a `CsVec` can represent the row of a CSR matrix without any copying.
///
/// Similar to [`CsMat`] and [`TriMat`], the `CsVecBase` type is parameterized
/// over the indexing storage backend `IStorage` and the data storage backend
/// `DStorage`. Type aliases are provided for common cases: [`CsVec`] represents
/// a sparse vector owning its data, with `Vec`s as storage backends;
/// [`CsVecView`] represents a sparse vector borrowing its data, using slices
/// as storage backends; and [`CsVecViewMut`] represents a sparse vector that
/// mutably borrows its data (but immutably borrows its indices).
///
/// Additionaly, the type aliases [`CsVecI`], [`CsVecViewI`], and
/// [`CsVecViewMutI`] can be used to choose an index type different from the
/// default `usize`.
///
/// [`CsMat`]: struct.CsMatBase.html
/// [`TriMat`]: struct.TriMatBase.html
/// [`CsVec`]: type.CsVec.html
/// [`CsVecView`]: type.CsVecView.html
/// [`CsVecViewMut`]: type.CsVecViewMut.html
/// [`CsVecI`]: type.CsVecI.html
/// [`CsVecViewI`]: type.CsVecViewI.html
/// [`CsVecViewMutI`]: type.CsVecViewMutI.html
#[derive(PartialEq, Debug, Clone)]
#[cfg_attr(
    all(feature = "serde", feature = "serde_derive"),
    derive(Serialize, Deserialize)
)]
pub struct CsVecBase<IStorage, DStorage> {
    dim: usize,
    indices: IStorage,
    data: DStorage,
}

pub type CsVecI<N, I> = CsVecBase<Vec<I>, Vec<N>>;
pub type CsVecViewI<'a, N, I> = CsVecBase<&'a [I], &'a [N]>;
pub type CsVecViewMutI<'a, N, I> = CsVecBase<&'a [I], &'a mut [N]>;

pub type CsVecView<'a, N> = CsVecViewI<'a, N, usize>;
pub type CsVecViewMut<'a, N> = CsVecViewMutI<'a, N, usize>;
pub type CsVec<N> = CsVecI<N, usize>;

/// Sparse matrix in the triplet format.
///
/// Sparse matrices in the triplet format use three arrays of equal sizes (accessible through the
/// methods [`row_inds`], [`col_inds`], [`data`]), the first one
/// storing the row indices of non-zero values, the second storing the
/// corresponding column indices and the last array storing the corresponding
/// scalar value. If a non-zero location is repeated in the arrays, the
/// non-zero value is taken as the sum of the corresponding scalar entries.
///
/// [`row_inds`]: struct.TriMatBase.html#method.row_inds
/// [`col_inds`]: struct.TriMatBase.html#method.col_inds
/// [`data`]: struct.TriMatBase.html#method.data
///
/// This format is useful for iteratively building a sparse matrix, since the
/// various non-zero entries can be specified in any order, or even partially
/// as is common in physics with partial derivatives equations.
///
/// This format cannot be used for arithmetic operations. Arithmetic operations
/// are more efficient in the [compressed format](struct.CsMatBase.html).
/// A matrix in the triplet format can be converted to the compressed format
/// using the methods [`to_csc`] and [`to_csr`].
///
/// [`to_csc`]: struct.TriMatBase.html#method.to_csc
/// [`to_csr`]: struct.TriMatBase.html#method.to_csr
///
/// The `TriMatBase` type is parameterized by the storage type for the row and
/// column indices, `IStorage`, and by the storage type for the non-zero values
/// `DStorage`. Convenient aliases are availaible to specify frequent variant:
/// [`TriMat`] refers to a triplet matrix owning the storage of its indices and
/// and values, [`TriMatView`] refers to a triplet matrix with slices to store
/// its indices and values, while [`TriMatViewMut`] refers to a a triplet matrix
/// using mutable slices.
///
/// Additionaly, the type aliases [`TriMatI`], [`TriMatViewI`] and
/// [`TriMatViewMutI`] can be used to choose an index type different from the
/// default `usize`.
///
/// [`TriMat`]: type.TriMat.html
/// [`TriMatView`]: type.TriMatView.html
/// [`TriMatViewMut`]: type.TriMatViewMut.html
/// [`TriMatI`]: type.TriMatI.html
/// [`TriMatViewI`]: type.TriMatViewI.html
/// [`TriMatViewMutI`]: type.TriMatViewMutI.html
#[derive(PartialEq, Debug)]
pub struct TriMatBase<IStorage, DStorage> {
    rows: usize,
    cols: usize,
    row_inds: IStorage,
    col_inds: IStorage,
    data: DStorage,
}

pub type TriMatI<N, I> = TriMatBase<Vec<I>, Vec<N>>;
pub type TriMatViewI<'a, N, I> = TriMatBase<&'a [I], &'a [N]>;
pub type TriMatViewMutI<'a, N, I> = TriMatBase<&'a mut [I], &'a mut [N]>;

pub type TriMat<N> = TriMatI<N, usize>;
pub type TriMatView<'a, N> = TriMatViewI<'a, N, usize>;
pub type TriMatViewMut<'a, N> = TriMatViewMutI<'a, N, usize>;

/// An iterator over elements of a sparse matrix, in the triplet format
///
/// The dataypes RI, CI, and DI are iterators yielding the row, column and
/// values of non-zero entries.
///
/// As in `TriMat`, no order guarantee is provided and the same location can
/// appear multiple times. The non-zero value is then considered as the sum
/// of all the entries sharing its location.
#[derive(PartialEq, Debug, Clone)]
pub struct TriMatIter<RI, CI, DI> {
    rows: usize,
    cols: usize,
    nnz: usize,
    row_inds: RI,
    col_inds: CI,
    data: DI,
}

mod prelude {
    pub use super::{
        CsMat, CsMatBase, CsMatI, CsMatVecView, CsMatVecView_, CsMatView,
        CsMatViewI, CsMatViewMut, CsMatViewMutI, CsVec, CsVecBase, CsVecI,
        CsVecView, CsVecViewI, CsVecViewMut, CsVecViewMutI, SparseMat, TriMat,
        TriMatBase, TriMatI, TriMatIter, TriMatView, TriMatViewI,
        TriMatViewMut, TriMatViewMutI,
    };
}

/// A trait for common members of sparse matrices
pub trait SparseMat {
    /// The number of rows of this matrix
    fn rows(&self) -> usize;

    /// The number of columns of this matrix
    fn cols(&self) -> usize;

    /// The number of nonzeros of this matrix
    fn nnz(&self) -> usize;
}

mod utils {
    use indexing::SpIndex;

    pub fn sort_indices_data_slices<N: Copy, I: SpIndex>(
        indices: &mut [I],
        data: &mut [N],
        buf: &mut Vec<(I, N)>,
    ) {
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

pub mod binop;
pub mod compressed;
pub mod construct;
pub mod csmat;
pub mod linalg;
pub mod permutation;
pub mod prod;
pub mod symmetric;
pub mod to_dense;
pub mod triplet;
pub mod triplet_iter;
pub mod vec;
