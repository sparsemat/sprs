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

#[derive(Deserialize)]
pub struct CsMatBaseShadow<N, I, IptrStorage, IndStorage, DataStorage, Iptr = I>
where
    I: SpIndex,
    Iptr: SpIndex,
    IptrStorage: Deref<Target = [Iptr]>,
    IndStorage: Deref<Target = [I]>,
    DataStorage: Deref<Target = [N]>,
{
    storage: CompressedStorage,
    nrows: usize,
    ncols: usize,
    indptr: IptrStorage,
    indices: IndStorage,
    data: DataStorage,
}

impl<IptrStorage, IndStorage, DStorage, N, I: SpIndex, Iptr: SpIndex>
    TryFrom<CsMatBaseShadow<N, I, IptrStorage, IndStorage, DStorage, Iptr>>
    for CsMatBase<N, I, IptrStorage, IndStorage, DStorage, Iptr>
where
    IndStorage: Deref<Target = [I]>,
    IptrStorage: Deref<Target = [Iptr]>,
    DStorage: Deref<Target = [N]>,
{
    type Error = SprsError;
    fn try_from(
        val: CsMatBaseShadow<N, I, IptrStorage, IndStorage, DStorage, Iptr>,
    ) -> Result<Self, Self::Error> {
        let CsMatBaseShadow {
            storage,
            nrows,
            ncols,
            indptr,
            indices,
            data,
        } = val;
        let shape = (nrows, ncols);
        Self::new_checked(storage, shape, indptr, indices, data)
            .map_err(|(_, _, _, e)| e)
    }
}

#[cfg(test)]
mod test {
    use crate::sparse::CsVecI;
    #[test]
    fn valid_vector() {
        let vec = CsVecI::new(5, vec![0_u16, 3, 4], vec![1_u8, 5, 6]);
        // Commenting this line and stuff works again...
        let json = serde_json::to_string(&vec).unwrap();
    }
}
