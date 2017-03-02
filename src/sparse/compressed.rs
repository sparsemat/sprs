///! Traits to generalize over compressed sparse matrices storages


use sparse::prelude::*;
use indexing::SpIndex;
use std::ops::{Deref};

/// The SpMatView trait describes data that can be seen as a view
/// into a CsMat
pub trait SpMatView<N, I: SpIndex> {
    /// Return a view into the current matrix
    fn view(&self) -> CsMatViewI<N, I>;

    /// Return a view into the current matrix
    fn transpose_view(&self) -> CsMatViewI<N, I>;
}


impl<N, I, IpStorage, IndStorage, DataStorage> SpMatView<N, I>
for CsMat<N, I, IpStorage, IndStorage, DataStorage>
where I: SpIndex,
      IpStorage: Deref<Target=[I]>,
      IndStorage: Deref<Target=[I]>,
      DataStorage: Deref<Target=[N]> {

    fn view(&self) -> CsMatViewI<N, I> {
        self.view()
    }

    fn transpose_view(&self) -> CsMatViewI<N, I> {
        self.transpose_view()
    }
}

/// The SpVecView trait describes types that can be seen as a view into
/// a CsVec
pub trait SpVecView<N> {
    /// Return a view into the current vector
    fn view(&self) ->  CsVecView<N>;
}

impl<N, IndStorage, DataStorage> SpVecView<N>
for CsVec<N, IndStorage, DataStorage>
where IndStorage: Deref<Target=[usize]>,
      DataStorage: Deref<Target=[N]> {

    fn view(&self) -> CsVecView<N> {
        self.view()
    }
}
