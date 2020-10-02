use super::*;
pub(crate) use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

#[derive(Deserialize)]
pub(crate) struct CsVecBaseShadow<IStorage, DStorage, N, I: SpIndex = usize>
where
    IStorage: Deref<Target = [I]>,
    DStorage: Deref<Target = [N]>,
{
    dim: usize,
    indices: IStorage,
    data: DStorage,
}

impl<IStorage, DStorage, N, I: SpIndex>
    TryFrom<CsVecBaseShadow<IStorage, DStorage, N, I>>
    for CsVecBase<IStorage, DStorage, N, I>
where
    IStorage: Deref<Target = [I]>,
    DStorage: Deref<Target = [N]>,
{
    type Error = SprsError;
    fn try_from(
        val: CsVecBaseShadow<IStorage, DStorage, N, I>,
    ) -> Result<Self, Self::Error> {
        let CsVecBaseShadow { dim, indices, data } = val;
        Self::new_(dim, indices, data).map_err(|(_, _, e)| e)
    }
}
