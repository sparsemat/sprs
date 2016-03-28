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

mod utils {
    pub fn sort_indices_data_slices<N: Copy>(indices: &mut [usize],
                                             data: &mut [N],
                                             buf: &mut Vec<(usize, N)>) {
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
