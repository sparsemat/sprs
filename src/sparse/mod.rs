use std::ops::Deref;

pub use self::csmat::{CompressedStorage};

pub use self::vec::{CsVec,
                    CsVecOwned,
                    CsVecView,
};

/// Compressed matrix in the CSR or CSC format.
#[derive(PartialEq, Debug)]
pub struct CsMat<N, IptrStorage, IndStorage, DataStorage>
where IptrStorage: Deref<Target=[usize]>,
      IndStorage: Deref<Target=[usize]>,
      DataStorage: Deref<Target=[N]> {
    storage: CompressedStorage,
    nrows : usize,
    ncols : usize,
    indptr : IptrStorage,
    indices : IndStorage,
    data : DataStorage
}

pub type CsMatOwned<N> = CsMat<N, Vec<usize>, Vec<usize>, Vec<N>>;
pub type CsMatView<'a, N> = CsMat<N, &'a [usize], &'a [usize], &'a [N]>;
pub type CsMatViewMut<'a, N> = CsMat<N, &'a [usize], &'a [usize], &'a mut [N]>;
// FIXME: a fixed size array would be better, but no Deref impl
pub type CsMatVecView<'a, N> = CsMat<N, Vec<usize>, &'a [usize], &'a [N]>;

mod prelude {
    pub use super::{
        CsMat,
        CsMatView,
        CsMatViewMut,
        CsMatOwned,
        CsMatVecView,
    };
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
