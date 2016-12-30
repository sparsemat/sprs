use std::ops::Deref;
use indexing::SpIndex;

pub use self::csmat::{CompressedStorage};

/// Compressed matrix in the CSR or CSC format.
#[derive(PartialEq, Debug)]
pub struct CsMat<N, I, IptrStorage, IndStorage, DataStorage>
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

pub type CsMatOwned_<N, I> = CsMat<N, I, Vec<I>, Vec<I>, Vec<N>>;
pub type CsMatView_<'a, N, I> = CsMat<N, I, &'a [I], &'a [I], &'a [N]>;
pub type CsMatViewMut_<'a, N, I> = CsMat<N, I, &'a [I], &'a [I], &'a mut [N]>;
pub type CsMatVecView_<'a, N, I> = CsMat<N, I, Vec<I>, &'a [I], &'a [N]>;

pub type CsMatOwned<N> = CsMatOwned_<N, usize>;
pub type CsMatView<'a, N> = CsMatView_<'a, N, usize>;
pub type CsMatViewMut<'a, N> = CsMatViewMut_<'a, N, usize>;
// FIXME: a fixed size array would be better, but no Deref impl
pub type CsMatVecView<'a, N> = CsMatVecView_<'a, N, usize>;

/// A sparse vector, storing the indices of its non-zero data.
/// The indices should be sorted.
#[derive(PartialEq, Debug)]
pub struct CsVec<N, IStorage, DStorage>
where DStorage: Deref<Target=[N]> {
    dim: usize,
    indices : IStorage,
    data : DStorage
}

pub type CsVecView_<'a, N, I> = CsVec<N, &'a [I], &'a [N]>;
pub type CsVecViewMut_<'a, N, I> = CsVec<N, &'a [I], &'a mut [N]>;
pub type CsVecOwned_<N, I> = CsVec<N, Vec<I>, Vec<N>>;

pub type CsVecView<'a, N> = CsVecView_<'a, N, usize>;
pub type CsVecViewMut<'a, N> = CsVecViewMut_<'a, N, usize>;
pub type CsVecOwned<N> = CsVecOwned_<N, usize>;

mod prelude {
    pub use super::{
        CsMat,
        CsMatView_,
        CsMatView,
        CsMatViewMut_,
        CsMatViewMut,
        CsMatOwned_,
        CsMatOwned,
        CsMatVecView_,
        CsMatVecView,
        CsVec,
        CsVecView_,
        CsVecView,
        CsVecViewMut_,
        CsVecViewMut,
        CsVecOwned_,
        CsVecOwned,
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
