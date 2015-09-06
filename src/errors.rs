//! Error type for sprs

#[derive(PartialEq, Debug)]
pub enum SprsError {
    IncompatibleDimensions,
    BadWorkspaceDimensions,
    IncompatibleStorages,
    BadStorageType,
    EmptyStackingList,
    NotImplemented,
    EmptyBmatRow,
    EmptyBmatCol,
    NonSortedIndices,
    OutOfBoundsIndex,
    BadIndptrLength,
    DataIndicesMismatch,
    BadNnzCount,
    OutOfBoundsIndptr,
    UnsortedIndptr,
}
