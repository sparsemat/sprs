//! Traits to generalize over compressed sparse matrices storages
use crate::indexing::SpIndex;
use crate::sparse::prelude::*;
use std::ops::Deref;

/// The `SpMatView` trait describes data that can be seen as a view
/// into a `CsMat`
pub trait SpMatView<N, I: SpIndex, Iptr: SpIndex = I> {
    /// Return a view into the current matrix
    fn view(&self) -> CsMatViewI<'_, N, I, Iptr>;

    /// Return a view into the current matrix
    fn transpose_view(&self) -> CsMatViewI<'_, N, I, Iptr>;
}

impl<N, I, Iptr, IpStorage, IndStorage, DataStorage> SpMatView<N, I, Iptr>
    for CsMatBase<N, I, IpStorage, IndStorage, DataStorage, Iptr>
where
    I: SpIndex,
    Iptr: SpIndex,
    IpStorage: Deref<Target = [Iptr]>,
    IndStorage: Deref<Target = [I]>,
    DataStorage: Deref<Target = [N]>,
{
    fn view(&self) -> CsMatViewI<'_, N, I, Iptr> {
        self.view()
    }

    fn transpose_view(&self) -> CsMatViewI<'_, N, I, Iptr> {
        self.transpose_view()
    }
}
