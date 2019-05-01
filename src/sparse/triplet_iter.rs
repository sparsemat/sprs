//! A structure for iterating over the non-zero values of any kind of
//! sparse matrix.

use num_traits::Num;

use indexing::SpIndex;
use sparse::csmat;
use sparse::{CsMatI, TriMatIter};

impl<'a, N, I, RI, CI, DI> Iterator for TriMatIter<RI, CI, DI>
where
    I: 'a + SpIndex,
    N: 'a,
    RI: Iterator<Item = &'a I>,
    CI: Iterator<Item = &'a I>,
    DI: Iterator<Item = &'a N>,
{
    type Item = (&'a N, (I, I));

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match (self.row_inds.next(), self.col_inds.next(), self.data.next()) {
            (Some(row), Some(col), Some(val)) => Some((val, (*row, *col))),
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.row_inds.size_hint() // FIXME merge hints?
    }
}

impl<'a, N, I, RI, CI, DI> TriMatIter<RI, CI, DI>
where
    I: 'a + SpIndex,
    N: 'a,
    RI: Iterator<Item = &'a I>,
    CI: Iterator<Item = &'a I>,
    DI: Iterator<Item = &'a N>,
{
    /// Create a new `TriMatIter` from iterators
    pub fn new(
        shape: (usize, usize),
        nnz: usize,
        row_inds: RI,
        col_inds: CI,
        data: DI,
    ) -> Self {
        Self {
            rows: shape.0,
            cols: shape.1,
            nnz,
            row_inds,
            col_inds,
            data,
        }
    }

    /// The number of rows of the matrix
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of cols of the matrix
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The shape of the matrix, as a `(rows, cols)` tuple
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    pub fn into_row_inds(self) -> RI {
        self.row_inds
    }

    pub fn into_col_inds(self) -> CI {
        self.col_inds
    }

    pub fn into_data(self) -> DI {
        self.data
    }

    pub fn transpose_into(self) -> TriMatIter<CI, RI, DI> {
        TriMatIter {
            rows: self.cols,
            cols: self.rows,
            nnz: self.nnz,
            row_inds: self.col_inds,
            col_inds: self.row_inds,
            data: self.data,
        }
    }
}

impl<'a, N, I, RI, CI, DI> TriMatIter<RI, CI, DI>
where
    I: 'a + SpIndex,
    N: 'a + Clone,
    RI: Clone + Iterator<Item = &'a I>,
    CI: Clone + Iterator<Item = &'a I>,
    DI: Clone + Iterator<Item = &'a N>,
{
    /// Consume this matrix to create a CSC matrix
    pub fn into_csc(self) -> CsMatI<N, I>
    where
        N: Num,
    {
        let mut row_counts = vec![I::zero(); self.rows() + 1];
        for i in self.clone().into_row_inds() {
            row_counts[i.index() + 1] += I::one();
        }
        let mut indptr = row_counts.clone();
        // cum sum
        for i in 1..=self.rows() {
            indptr[i] = indptr[i] + indptr[i - 1];
        }
        let nnz_max = indptr[self.rows()].index();
        let mut indices = vec![I::zero(); nnz_max];
        let mut data = vec![N::zero(); nnz_max];

        // reset row counts to 0
        for count in row_counts.iter_mut() {
            *count = I::zero();
        }

        for (val, (i, j)) in self.clone() {
            let i = i.index();
            let j = j.index();
            let start = indptr[i].index();
            let stop = start + row_counts[i].index();
            let col_exists = {
                let mut col_exists = false;
                let iter = indices[start..stop]
                    .iter()
                    .zip(data[start..stop].iter_mut());
                for (&col_cell, data_cell) in iter {
                    if col_cell.index() == j {
                        *data_cell = data_cell.clone() + val.clone();
                        col_exists = true;
                        break;
                    }
                }
                col_exists
            };
            if !col_exists {
                indices[stop] = I::from_usize(j);
                data[stop] = val.clone();
                row_counts[i] += I::one();
            }
        }

        // compress the nonzero entries
        let mut dst_start = indptr[0].index();
        for i in 0..self.rows() {
            let start = indptr[i].index();
            let col_nnz = row_counts[i].index();
            if start != dst_start {
                for k in 0..col_nnz {
                    indices[dst_start + k] = indices[start + k];
                    data[dst_start + k] = data[start + k].clone();
                }
            }
            indptr[i] = I::from_usize(dst_start);
            dst_start += col_nnz;
        }
        indptr[self.rows()] = I::from_usize(dst_start);

        // at this point we have a CSR matrix with unsorted columns
        // transposing it will yield the desired CSC matrix with sorted rows
        let nnz = indptr[self.rows()].index();
        let mut out_indptr = vec![I::zero(); self.cols() + 1];
        let mut out_indices = vec![I::zero(); nnz];
        let mut out_data = vec![N::zero(); nnz];
        csmat::raw::convert_storage(
            csmat::CompressedStorage::CSR,
            self.shape(),
            &indptr,
            &indices[..nnz],
            &data[..nnz],
            &mut out_indptr,
            &mut out_indices,
            &mut out_data,
        );
        CsMatI {
            storage: csmat::CompressedStorage::CSC,
            nrows: self.rows,
            ncols: self.cols,
            indptr: out_indptr,
            indices: out_indices,
            data: out_data,
        }
    }
}
