///! Traits to generalize over compressed sparse matrices storages


use sparse::csmat::{CsMat, CsMatView};
use std::ops::{Deref};

/// The SpMatView trait describes data that can be seen as a view
/// into a CsMat
pub trait SpMatView<N> {
    /// Return a view into the current matrix
    fn borrowed(&self) -> CsMatView<N>;

    /// Return a view into the current matrix
    fn transpose_view(&self) -> CsMatView<N>;
}


impl<N, IpStorage, IndStorage, DataStorage> SpMatView<N>
for CsMat<N, IpStorage, IndStorage, DataStorage>
where N: Copy,
      IpStorage: Deref<Target=[usize]>,
      IndStorage: Deref<Target=[usize]>,
      DataStorage: Deref<Target=[N]> {

    fn borrowed(&self) -> CsMatView<N> {
        self.borrowed()
    }

    fn transpose_view(&self) -> CsMatView<N> {
        self.transpose_view()
    }
}

