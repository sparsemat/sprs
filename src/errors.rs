//! Error type for sprs

use std::error::Error;
use std::fmt;

#[derive(PartialEq, Debug)]
pub enum SprsError {
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
