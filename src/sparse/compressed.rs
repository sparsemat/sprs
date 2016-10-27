///! Traits to generalize over compressed sparse matrices storages


use sparse::prelude::*;
use std::ops::{Deref};

/// The SpMatView trait describes data that can be seen as a view
/// into a CsMat
pub trait SpMatView<N> {
    /// Return a view into the current matrix
    fn view(&self) -> CsMatView<N>;

    /// Return a view into the current matrix
    fn transpose_view(&self) -> CsMatView<N>;
}


impl<N, IpStorage, IndStorage, DataStorage> SpMatView<N>
for CsMat<N, IpStorage, IndStorage, DataStorage>
where IpStorage: Deref<Target=[usize]>,
      IndStorage: Deref<Target=[usize]>,
      DataStorage: Deref<Target=[N]> {

    fn view(&self) -> CsMatView<N> {
        self.view()
    }

    fn transpose_view(&self) -> CsMatView<N> {
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
