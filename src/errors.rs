//! Error type for sprs

use std::error::Error;
use std::fmt;

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
    TooLargeIndptr,
    UnsortedIndptr,
    EmptyBlock,
    SingularMatrix,
    NonSquareMatrix,
}

use self::SprsError::*;

impl SprsError {
    fn descr(&self) -> &str {
        match *self {
            IncompatibleDimensions => "matrices dimensions do not agree",
            BadWorkspaceDimensions =>
                "workspace dimension does not match requirements",
            IncompatibleStorages => "incompatible storages",
            BadStorageType => "wrong storage type",
            EmptyStackingList => "stacking list is empty",
            NotImplemented => "this method is not yet implemented",
            EmptyBmatRow => "empty row in bmat argument",
            EmptyBmatCol => "empty column in bmat argument",
            NonSortedIndices => "a vector's indices are not sorted",
            OutOfBoundsIndex => "an element in indices is out of bounds",
            BadIndptrLength => "inpdtr's length doesn't agree with dimensions",
            DataIndicesMismatch => "data and indices lengths differ",
            BadNnzCount => "the nnz count and indptr do not agree",
            OutOfBoundsIndptr => "some indptr values are out of bounds",
            TooLargeIndptr => "indptr value > usize::MAX / 2",
            UnsortedIndptr => "indptr is not sorted",
            EmptyBlock => "tried to create an empty block",
            SingularMatrix => "matrix is singular",
            NonSquareMatrix => "matrix should be square",
        }
    }
}

impl Error for SprsError {
    fn description(&self) -> &str {
        self.descr()
    }
}

impl fmt::Display for SprsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.descr().fmt(f)
    }
}
