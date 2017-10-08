//! Serialization and deserialization of sparse matrices

use std::path::Path;
use std::io;
use std::io::BufRead;
use std::fs::File;
use std::error::Error;
use std::fmt;

use num_traits::cast::NumCast;

use sparse::TriMatI;
use indexing::SpIndex;

#[derive(Debug)]
pub enum IoError {
    Io(io::Error),
    BadMatrixMarketFile,
    UnsupportedMatrixMarketFormat,
}

use self::IoError::*;

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            IoError::Io(ref err) => err.fmt(f),
            IoError::BadMatrixMarketFile =>
                write!(f, "Bad matrix market file."),
            IoError::UnsupportedMatrixMarketFormat =>
                write!(f, "Bad matrix market file."),
        }
    }
}

impl Error for IoError {
    fn description(&self) -> &str {
        match *self {
            IoError::Io(ref err) => err.description(),
            IoError::BadMatrixMarketFile => "bad matrix market file",
            IoError::UnsupportedMatrixMarketFormat => "unsupported format",
        }
    }

    fn cause(&self) -> Option<&Error> {
        match *self {
            IoError::Io(ref err) => Some(err),
            IoError::BadMatrixMarketFile => None,
            IoError::UnsupportedMatrixMarketFormat => None,
        }
    }
}

impl From<io::Error> for IoError {
    fn from(err: io::Error) -> IoError {
        IoError::Io(err)
    }
}

enum DataType {
    Integer,
    Real,
}

/// Read a sparse matrix file in the Matrix Market format and return a
/// corresponding triplet matrix.
///
/// Presently, only general matrices are supported, but symmetric and hermitian
/// matrices should be supported in the future.
pub fn read_matrix_market<N, I, P>(mm_file: P) -> Result<TriMatI<N, I>, IoError>
where I: SpIndex,
      N: NumCast,
      P: AsRef<Path>,
{
    let mm_file = mm_file.as_ref();
    let f = File::open(mm_file)?;
    let mut reader = io::BufReader::new(f);
    // MatrixMarket format specifies lines of at most 1024 chars
    let mut line = String::with_capacity(1024);

    // Parse the header line, all tags are case insensitive.
    reader.read_line(&mut line)?;
    let header = line.to_lowercase();
    if !header.starts_with("%%matrixmarket matrix coordinate") {
        return Err(BadMatrixMarketFile);
    }
    if !header.contains("general") {
        return Err(UnsupportedMatrixMarketFormat);
    }
    let data_type = if line.contains("real") {
        DataType::Real
    } else if line.contains("integer") {
        DataType::Integer
    } else {
        // we currently don't support complex
        return Err(UnsupportedMatrixMarketFormat);
    };
    // The header is followed by any number of comment or empty lines, skip
    loop {
        line.clear();
        let len = reader.read_line(&mut line)?;
        if len == 0 || line.starts_with("%") {
            continue;
        } else {
            break;
        }
    }
    // read shape and number of entries
    // this is a line like:
    // rows cols entries
    // with arbitrary amounts of whitespace
    let (rows, cols, entries) = {
        let mut infos = line.split_whitespace()
                            .filter_map(|s| s.parse::<usize>().ok());
        let rows = infos.next().ok_or(BadMatrixMarketFile)?;
        let cols = infos.next().ok_or(BadMatrixMarketFile)?;
        let entries = infos.next().ok_or(BadMatrixMarketFile)?;
        if infos.next().is_some() {
            return Err(BadMatrixMarketFile);
        }
        (rows, cols, entries)
    };
    let mut row_inds = Vec::with_capacity(entries);
    let mut col_inds = Vec::with_capacity(entries);
    let mut data = Vec::with_capacity(entries);
    // one non-zero entry per non-empty line
    for _ in 0..entries {
        // skip empty lines (no comment line should appear)
        loop {
            line.clear();
            let len = reader.read_line(&mut line)?;
            if len == 0 {
                continue;
            } else {
                break;
            }
        }
        // Non-zero entries are lines of the form:
        // row col value
        // if the data type is integer of real, and
        // row col real imag
        // if the data type is complex.
        // Again, this is with arbitrary amounts of whitespace
        let mut entry = line.split_whitespace();
        let row = entry.next()
                       .ok_or(BadMatrixMarketFile)
                       .and_then(|s| s.parse::<usize>()
                                      .or(Err(BadMatrixMarketFile)))?;
        let col = entry.next()
                       .ok_or(BadMatrixMarketFile)
                       .and_then(|s| s.parse::<usize>()
                                      .or(Err(BadMatrixMarketFile)))?;
        // MatrixMarket indices are 1-based
        let row = row.checked_sub(1).ok_or(BadMatrixMarketFile)?;
        let col = col.checked_sub(1).ok_or(BadMatrixMarketFile)?;
        row_inds.push(I::from_usize(row));
        col_inds.push(I::from_usize(col));
        match data_type {
            DataType::Integer => {
                let val = entry.next()
                               .ok_or(BadMatrixMarketFile)
                               .and_then(|s| s.parse::<usize>()
                                              .or(Err(BadMatrixMarketFile)))?;
                data.push(NumCast::from(val).unwrap());
            },
            DataType::Real => {
                let val = entry.next()
                               .ok_or(BadMatrixMarketFile)
                               .and_then(|s| s.parse::<f64>()
                                              .or(Err(BadMatrixMarketFile)))?;
                data.push(NumCast::from(val).unwrap());
            },
        }
        if entry.next().is_some() {
            return Err(BadMatrixMarketFile);
        }
    }

    Ok(TriMatI::from_triplets((rows, cols), row_inds, col_inds, data))
}

#[cfg(test)]
mod test {
    use super::read_matrix_market;
    #[test]
    fn simple_matrix_market_read() {
        let path = "data/matrix_market/simple.mm";
        let mat = read_matrix_market::<f64, usize, _>(path).unwrap();
        assert_eq!(mat.rows(), 5);
        assert_eq!(mat.cols(), 5);
        assert_eq!(mat.nnz(), 8);
        assert_eq!(mat.row_inds(), &[0, 1, 2, 0, 3, 3, 3, 4]);
        assert_eq!(mat.col_inds(), &[0, 1, 2, 3, 1, 3, 4, 4]);
        assert_eq!(mat.data(),
                   &[1., 10.5, 1.5e-02, 6., 2.505e2, -2.8e2, 3.332e1, 1.2e+1]);
    }
}
