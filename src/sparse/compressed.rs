///! Traits to generalize over compressed sparse matrices storages


use sparse::csmat::{CsMat, CsMatVec, CsMatView};
use sparse::binop;
use num::traits::Num;
use std::ops::{Deref, Add};

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

impl<'a, 'b, N, IStorage, DStorage, Mat> Add<&'b Mat>
for &'a CsMat<N, IStorage, DStorage>
where N: 'a + Copy + Num + Default,
      IStorage: 'a + Deref<Target=[usize]>,
      DStorage: 'a + Deref<Target=[N]>,
      Mat: SpMatView<N> {
    type Output = CsMatVec<N>;

    fn add(self, rhs: &'b Mat) -> CsMatVec<N> {
        if self.storage() != rhs.borrowed().storage() {
            return binop::add_mat_same_storage(
                self, &rhs.borrowed().to_other_storage()).unwrap()
        }
        binop::add_mat_same_storage(self, rhs).unwrap()
    }
}
