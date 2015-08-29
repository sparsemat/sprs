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
}
