//! Error type for sprs

use std::error::Error;
use std::fmt;

#[derive(PartialEq, Debug)]
pub enum SprsError {
    NonSortedIndices,
    UnsortedIndptr,
    SingularMatrix,
    IllegalArguments(&'static str),
}

use self::SprsError::*;

impl SprsError {
    fn descr(&self) -> &str {
        match *self {
            NonSortedIndices => "a vector's indices are not sorted",
            UnsortedIndptr => "indptr is not sorted",
            SingularMatrix => "matrix is singular",
            IllegalArguments(s) => s,
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
