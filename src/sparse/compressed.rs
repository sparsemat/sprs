///! Traits to generalize over compressed sparse matrices storages


use sparse::csmat::{CsMat, CsMatVec, CsMatView};
use std::ops::{Deref};

/// The SpMatView trait describes data that can be seen as a view
/// into a CsMat
pub trait SpMatView<N> {
    /// Return a view into the current matrix
    fn borrowed(&self) -> CsMatView<N>;
}


impl<N, IndStorage, DataStorage> SpMatView<N>
for CsMat<N, IndStorage, DataStorage>
where N: Copy,
      IndStorage: Deref<Target=[usize]>,
      DataStorage: Deref<Target=[N]> {

    fn borrowed(&self) -> CsMatView<N> {
        self.borrowed()
    }
}

