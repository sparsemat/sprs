//! Error type for sprs

#[derive(PartialEq, Debug)]
pub enum SprsError {
    IncompatibleDimensions,
    IncompatibleStorages,
    EmptyStackingList,
    NotImplemented,
    EmptyBmatRow,
    EmptyBmatCol,
}
