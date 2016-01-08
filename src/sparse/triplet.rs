///! Triplet format matrix
///! Useful for building a matrix, but not for computations

use sparse::csmat;
use num::traits::Num;

/// Indexing type into a Triplet
pub struct TripletIndex(pub usize);

/// Triplet matrix
pub struct TripletMat<N> {
    rows: usize,
    cols: usize,
    row_inds: Vec<usize>,
    col_inds: Vec<usize>,
    data: Vec<N>,
}

impl<N> TripletMat<N> {

    pub fn new(shape: (usize, usize)) -> TripletMat<N> {
        TripletMat {
            rows: shape.0,
            cols: shape.1,
            row_inds: Vec::new(),
            col_inds: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn with_capacity(shape: (usize, usize), cap: usize) -> TripletMat<N> {
        TripletMat {
            rows: shape.0,
            cols: shape.1,
            row_inds: Vec::with_capacity(cap),
            col_inds: Vec::with_capacity(cap),
            data: Vec::with_capacity(cap),
        }
    }

    pub fn from_triplets(shape: (usize, usize),
                         row_inds: Vec<usize>,
                         col_inds: Vec<usize>,
                         data: Vec<N>)
                         -> TripletMat<N> {
        assert!(row_inds.len() == col_inds.len(),
                "all inputs should have the same length");
        assert!(data.len() == col_inds.len(),
                "all inputs should have the same length");
        assert!(row_inds.len() == data.len(),
                "all inputs should have the same length");
        assert!(row_inds.iter().all(|&i| i < shape.0),
                "row indices should be within shape");
        assert!(col_inds.iter().all(|&j| j < shape.1),
                "col indices should be within shape");
        TripletMat {
            rows: shape.0,
            cols: shape.1,
            row_inds: row_inds,
            col_inds: col_inds,
            data: data,
        }
    }

    pub fn rows(&self) -> usize {
        self.borrowed().rows()
    }

    pub fn cols(&self) -> usize {
        self.borrowed().cols()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.borrowed().shape()
    }

    pub fn nnz(&self) -> usize {
        self.borrowed().nnz()
    }

    pub fn row_inds(&self) -> &[usize] {
        self.borrowed().row_inds()
    }

    pub fn col_inds(&self) -> &[usize] {
        self.borrowed().col_inds()
    }

    pub fn data(&self) -> &[N] {
        self.borrowed().data()
    }

    pub fn find_locations(&self, row: usize, col: usize) -> Vec<TripletIndex> {
        self.borrowed().find_locations(row, col)
    }

    pub fn borrowed(&self) -> TripletView<N> {
        TripletView {
            rows: self.rows,
            cols: self.cols,
            row_inds: &self.row_inds[..],
            col_inds: &self.col_inds[..],
            data: &self.data[..],
        }
    }

    pub fn set_triplet(&mut self,
                       TripletIndex(triplet_ind): TripletIndex,
                       row: usize,
                       col: usize,
                       val: N) {
        self.borrowed_mut()
            .set_triplet(TripletIndex(triplet_ind), row, col, val);
    }

    pub fn borrowed_mut(&mut self) -> TripletViewMut<N> {
        TripletViewMut {
            rows: self.rows,
            cols: self.cols,
            row_inds: &mut self.row_inds[..],
            col_inds: &mut self.col_inds[..],
            data: &mut self.data[..],
        }
    }

    pub fn add_triplet(&mut self, row: usize, col: usize, val: N) {
        assert!(row < self.rows);
        assert!(col < self.cols);
        self.row_inds.push(row);
        self.col_inds.push(col);
        self.data.push(val);
    }

    pub fn reserve(&mut self, cap: usize) {
        self.row_inds.reserve(cap);
        self.col_inds.reserve(cap);
        self.data.reserve(cap);
    }

    pub fn reserve_exact(&mut self, cap: usize) {
        self.row_inds.reserve_exact(cap);
        self.col_inds.reserve_exact(cap);
        self.data.reserve_exact(cap);
    }

    pub fn to_csc(&self) -> csmat::CsMatOwned<N>
    where N: Copy + Num
    {
        self.borrowed().to_csc()
    }
}

/// Triplet matrix view
pub struct TripletView<'a, N: 'a> {
    rows: usize,
    cols: usize,
    row_inds: &'a [usize],
    col_inds: &'a [usize],
    data: &'a [N],
}

impl<'a, N> TripletView<'a, N> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn row_inds(&self) -> &'a [usize] {
        self.row_inds
    }

    pub fn col_inds(&self) -> &'a [usize] {
        self.col_inds
    }

    pub fn data(&self) -> &'a [N] {
        self.data
    }

    pub fn find_locations(&self, row: usize, col: usize) -> Vec<TripletIndex> {
        self.row_inds
            .iter()
            .zip(self.col_inds.iter())
            .enumerate()
            .filter(|&(_, (&i, &j))| i == row && j == col)
            .map(|(ind, _)| TripletIndex(ind))
            .collect()
    }

    pub fn to_csc(&self) -> csmat::CsMatOwned<N>
    where N: Copy + Num
    {
        let mut row_counts = vec![0; self.rows() + 1];
        for &i in self.row_inds.iter() {
            row_counts[i + 1] += 1;
        }
        let mut indptr = row_counts.clone();
        // cum sum
        for i in 1..(self.rows() + 1) {
            indptr[i] += indptr[i - 1];
        }
        let nnz_max = indptr[self.rows()];
        let mut indices = vec![0; nnz_max];
        let mut data = vec![N::zero(); nnz_max];

        // reset row counts to 0
        for mut count in row_counts.iter_mut() {
            *count = 0;
        }

        for (&val, (&i, &j)) in self.data
                                    .iter()
                                    .zip(self.row_inds
                                             .iter()
                                             .zip(self.col_inds.iter())) {
            let start = indptr[i];
            let stop = start + row_counts[i];
            let col_exists = {
                let mut col_exists = false;
                let iter = indices[start..stop]
                               .iter()
                               .zip(data[start..stop].iter_mut());
                for (&col_cell, mut data_cell) in iter {
                    if col_cell == j {
                        *data_cell = *data_cell + val;
                        col_exists = true;
                        break;
                    }
                }
                col_exists
            };
            if !col_exists {
                indices[stop] = j;
                data[stop] = val;
                row_counts[i] += 1;
            }
        }

        // compress the nonzero entries
        let mut dst_start = indptr[0];
        for i in 0..self.rows() {
            let start = indptr[i];
            let col_nnz = row_counts[i];
            if start != dst_start {
                for k in 0..col_nnz {
                    indices[dst_start + k] = indices[start + k];
                    data[dst_start + k] = data[start + k];
                }
            }
            indptr[i] = dst_start;
            dst_start += col_nnz;
        }

        // at this point we have a CSR matrix with unsorted columns
        // transposing it will yield the desired CSC matrix with sorted rows
        let nnz = indptr[self.rows()];
        let mut out_indptr = vec![0; self.cols() + 1];
        let mut out_indices = vec![0; nnz];
        let mut out_data = vec![N::zero(); nnz];
        csmat::raw::convert_storage(csmat::CompressedStorage::CSR,
                                    self.rows(),
                                    self.cols(),
                                    &indptr,
                                    &indices,
                                    &data,
                                    &mut out_indptr,
                                    &mut out_indices,
                                    &mut out_data);
        csmat::CsMatOwned::new_owned(csmat::CompressedStorage::CSC,
                                     self.rows,
                                     self.cols,
                                     out_indptr,
                                     out_indices,
                                     out_data
                                    ).expect("struct ensured by previous code")
    }
}


/// Triplet matrix mutable view
pub struct TripletViewMut<'a, N: 'a> {
    rows: usize,
    cols: usize,
    row_inds: &'a mut [usize],
    col_inds: &'a mut [usize],
    data: &'a mut [N],
}

impl<'a, N> TripletViewMut<'a, N> {

    pub fn rows(&self) -> usize {
        self.borrowed().rows()
    }

    pub fn cols(&self) -> usize {
        self.borrowed().cols()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.borrowed().shape()
    }

    pub fn nnz(&self) -> usize {
        self.borrowed().nnz()
    }

    pub fn row_inds(&self) -> &[usize] {
        self.borrowed().row_inds()
    }

    pub fn col_inds(&self) -> &[usize] {
        self.borrowed().col_inds()
    }

    pub fn data(&self) -> &[N] {
        self.borrowed().data()
    }

    pub fn borrowed(&self) -> TripletView<N> {
        TripletView {
            rows: self.rows,
            cols: self.cols,
            row_inds: &self.row_inds[..],
            col_inds: &self.col_inds[..],
            data: &self.data[..],
        }
    }

    pub fn set_triplet(&mut self,
                       TripletIndex(triplet_ind): TripletIndex,
                       row: usize,
                       col: usize,
                       val: N) {
        self.row_inds[triplet_ind] = row;
        self.col_inds[triplet_ind] = col;
        self.data[triplet_ind] = val;
    }

    pub fn to_csc(&self) -> csmat::CsMatOwned<N>
    where N: Copy + Num
    {
        self.borrowed().to_csc()
    }
}

#[cfg(test)]
mod test {

    use super::TripletMat;
    use sparse::csmat;
    use sparse::csmat::CompressedStorage::CSC;

    #[test]
    fn triplet_incremental() {
        let mut triplet_mat = TripletMat::with_capacity((4, 4), 6);
        // |1 2    |
        // |3      |
        // |      4|
        // |    5 6|
        triplet_mat.add_triplet(0, 0, 1.);
        triplet_mat.add_triplet(0, 1, 2.);
        triplet_mat.add_triplet(1, 0, 3.);
        triplet_mat.add_triplet(2, 3, 4.);
        triplet_mat.add_triplet(3, 2, 5.);
        triplet_mat.add_triplet(3, 3, 6.);

        let csc = triplet_mat.to_csc();
        let expected = csmat::CsMatOwned::new_owned(CSC,
                                                    4,
                                                    4,
                                                    vec![0, 2, 3, 4, 6],
                                                    vec![0, 1, 0, 3, 2, 3],
                                                    vec![1., 3., 2., 5., 4., 6.]
                                                    ).unwrap();
        assert_eq!(csc, expected);
    }

    #[test]
    fn triplet_unordered() {
        let mut triplet_mat = TripletMat::with_capacity((4, 4), 6);
        // |1 2    |
        // |3      |
        // |      4|
        // |    5 6|

        // the only difference with the triplet_incremental test is that
        // the triplets are added with non-sorted indices, therefore
        // testing the ability of the conversion to yield sorted output
        triplet_mat.add_triplet(0, 1, 2.);
        triplet_mat.add_triplet(0, 0, 1.);
        triplet_mat.add_triplet(1, 0, 3.);
        triplet_mat.add_triplet(2, 3, 4.);
        triplet_mat.add_triplet(3, 3, 6.);
        triplet_mat.add_triplet(3, 2, 5.);

        let csc = triplet_mat.to_csc();
        let expected = csmat::CsMatOwned::new_owned(CSC,
                                                    4,
                                                    4,
                                                    vec![0, 2, 3, 4, 6],
                                                    vec![0, 1, 0, 3, 2, 3],
                                                    vec![1., 3., 2., 5., 4., 6.]
                                                    ).unwrap();
        assert_eq!(csc, expected);
    }
}
