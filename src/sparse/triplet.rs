///! Triplet format matrix
///!
///! Useful for building a matrix, but not for computations. Therefore this
///! struct is mainly used to initialize a matrix before converting to
///! to a CsMat.
///!
///! A triplet format matrix is formed of three arrays of equal length, storing
///! the row indices, the column indices, and the values of the non-zero
///! entries. By convention, duplicate locations are summed up when converting
///! into CsMat.

use sparse::{csmat, CsMatI};
use num_traits::Num;
use indexing::SpIndex;

/// Indexing type into a Triplet
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TripletIndex(pub usize);

/// Triplet matrix owning its data
pub struct TriMatBase<N, I> {
    rows: usize,
    cols: usize,
    row_inds: Vec<I>,
    col_inds: Vec<I>,
    data: Vec<N>,
}

/// Triplet matrix view
pub struct TriMatViewBase<'a, N: 'a, I: 'a> {
    rows: usize,
    cols: usize,
    row_inds: &'a [I],
    col_inds: &'a [I],
    data: &'a [N],
}

/// Triplet matrix mutable view
pub struct TriMatViewMutBase<'a, N: 'a, I: 'a> {
    rows: usize,
    cols: usize,
    row_inds: &'a mut [I],
    col_inds: &'a mut [I],
    data: &'a mut [N],
}

pub type TriMat<N> = TriMatBase<N, usize>;
pub type TriMatView<'a, N> = TriMatViewBase<'a, N, usize>;
pub type TriMatViewMut<'a, N> = TriMatViewMutBase<'a, N, usize>;
pub type TriMatI<N, I> = TriMatBase<N, I>;
pub type TriMatViewI<'a, N, I> = TriMatViewBase<'a, N, I>;
pub type TriMatViewMutI<'a, N, I> = TriMatViewMutBase<'a, N, I>;


impl<N, I: SpIndex> TriMatBase<N, I> {
    /// Create a new triplet matrix of shape `(nb_rows, nb_cols)`
    pub fn new(shape: (usize, usize)) -> TriMatBase<N, I> {
        TriMatBase {
            rows: shape.0,
            cols: shape.1,
            row_inds: Vec::new(),
            col_inds: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Create a new triplet matrix of shape `(nb_rows, nb_cols)`, and
    /// pre-allocate `cap` elements on the backing storage
    pub fn with_capacity(shape: (usize, usize), cap: usize) -> TriMatBase<N, I> {
        TriMatBase {
            rows: shape.0,
            cols: shape.1,
            row_inds: Vec::with_capacity(cap),
            col_inds: Vec::with_capacity(cap),
            data: Vec::with_capacity(cap),
        }
    }

    /// Create a triplet matrix from its raw components. All arrays should
    /// have the same length.
    ///
    /// # Panics
    ///
    /// - if the arrays don't have the same length
    /// - if either the row or column indices are out of bounds.
    pub fn from_triplets(shape: (usize, usize),
                         row_inds: Vec<I>,
                         col_inds: Vec<I>,
                         data: Vec<N>)
                         -> TriMatBase<N, I> {
        assert!(row_inds.len() == col_inds.len(),
                "all inputs should have the same length");
        assert!(data.len() == col_inds.len(),
                "all inputs should have the same length");
        assert!(row_inds.len() == data.len(),
                "all inputs should have the same length");
        assert!(row_inds.iter().all(|&i| i.index() < shape.0),
                "row indices should be within shape");
        assert!(col_inds.iter().all(|&j| j.index() < shape.1),
                "col indices should be within shape");
        TriMatBase {
            rows: shape.0,
            cols: shape.1,
            row_inds: row_inds,
            col_inds: col_inds,
            data: data,
        }
    }

    /// The number of rows of the matrix
    pub fn rows(&self) -> usize {
        self.borrowed().rows()
    }

    /// The number of cols of the matrix
    pub fn cols(&self) -> usize {
        self.borrowed().cols()
    }

    /// The shape of the matrix, as a `(rows, cols)` tuple
    pub fn shape(&self) -> (usize, usize) {
        self.borrowed().shape()
    }

    /// The number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.borrowed().nnz()
    }

    /// The non-zero row indices
    pub fn row_inds(&self) -> &[I] {
        self.borrowed().row_inds()
    }

    /// The non-zero column indices
    pub fn col_inds(&self) -> &[I] {
        self.borrowed().col_inds()
    }

    /// The non-zero values
    pub fn data(&self) -> &[N] {
        self.borrowed().data()
    }

    /// Find all non-zero entries at the location given by `row` and `col`
    pub fn find_locations(&self, row: usize, col: usize) -> Vec<TripletIndex> {
        self.borrowed().find_locations(row, col)
    }

    /// Return a view of this matrix
    pub fn borrowed(&self) -> TriMatViewBase<N, I> {
        TriMatViewBase {
            rows: self.rows,
            cols: self.cols,
            row_inds: &self.row_inds[..],
            col_inds: &self.col_inds[..],
            data: &self.data[..],
        }
    }

    /// Replace a non-zero value at the given index.
    /// Indices can be obtained using find_locations.
    pub fn set_triplet(&mut self,
                       TripletIndex(triplet_ind): TripletIndex,
                       row: usize,
                       col: usize,
                       val: N) {
        self.borrowed_mut()
            .set_triplet(TripletIndex(triplet_ind), row, col, val);
    }

    /// Get a mutable view into this matrix.
    pub fn borrowed_mut(&mut self) -> TriMatViewMutBase<N, I> {
        TriMatViewMutBase {
            rows: self.rows,
            cols: self.cols,
            row_inds: &mut self.row_inds[..],
            col_inds: &mut self.col_inds[..],
            data: &mut self.data[..],
        }
    }

    /// Get a transposed view of this matrix
    pub fn transpose_view(&self) -> TriMatViewBase<N, I> {
        self.borrowed().transpose_view()
    }

    /// Append a non-zero triplet to this matrix.
    pub fn add_triplet(&mut self, row: usize, col: usize, val: N) {
        assert!(row < self.rows);
        assert!(col < self.cols);
        self.row_inds.push(I::from_usize(row));
        self.col_inds.push(I::from_usize(col));
        self.data.push(val);
    }

    /// Reserve `cap` additional non-zeros
    pub fn reserve(&mut self, cap: usize) {
        self.row_inds.reserve(cap);
        self.col_inds.reserve(cap);
        self.data.reserve(cap);
    }

    /// Reserve exactly `cap` non-zeros
    pub fn reserve_exact(&mut self, cap: usize) {
        self.row_inds.reserve_exact(cap);
        self.col_inds.reserve_exact(cap);
        self.data.reserve_exact(cap);
    }

    /// Create a CSC matrix from this triplet matrix
    pub fn to_csc(&self) -> CsMatI<N, I>
    where N: Clone + Num
    {
        self.borrowed().to_csc()
    }

    /// Create a CSR matrix from this triplet matrix
    pub fn to_csr(&self) -> CsMatI<N, I>
    where N: Clone + Num
    {
        self.borrowed().to_csr()
    }
}


impl<'a, N, I: SpIndex> TriMatViewBase<'a, N, I> {
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
        self.data.len()
    }

    /// The non-zero row indices
    pub fn row_inds(&self) -> &'a [I] {
        self.row_inds
    }

    /// The non-zero column indices
    pub fn col_inds(&self) -> &'a [I] {
        self.col_inds
    }

    /// The non-zero values
    pub fn data(&self) -> &'a [N] {
        self.data
    }

    /// Find all non-zero entries at the location given by `row` and `col`
    pub fn find_locations(&self, row: usize, col: usize) -> Vec<TripletIndex> {
        self.row_inds
            .iter()
            .zip(self.col_inds.iter())
            .enumerate()
            .filter(|&(_, (&i, &j))| i.index() == row && j.index() == col)
            .map(|(ind, _)| TripletIndex(ind))
            .collect()
    }

    /// Get a transposed view of this matrix
    pub fn transpose_view(&self) -> TriMatViewBase<'a, N, I> {
        TriMatViewBase {
            rows: self.cols,
            cols: self.rows,
            row_inds: self.col_inds,
            col_inds: self.row_inds,
            data: self.data,
        }
    }

    /// Create a CSC matrix from this triplet matrix
    pub fn to_csc(&self) -> CsMatI<N, I>
    where N: Clone + Num
    {
        let mut row_counts = vec![I::zero(); self.rows() + 1];
        for &i in self.row_inds.iter() {
            row_counts[i.index() + 1] += I::one();
        }
        let mut indptr = row_counts.clone();
        // cum sum
        for i in 1..(self.rows() + 1) {
            indptr[i] += indptr[i - 1];
        }
        let nnz_max = indptr[self.rows()].index();
        let mut indices = vec![I::zero(); nnz_max];
        let mut data = vec![N::zero(); nnz_max];

        // reset row counts to 0
        for mut count in row_counts.iter_mut() {
            *count = I::zero();
        }

        for (val, (&i, &j)) in self.data
                                   .iter()
                                   .zip(self.row_inds
                                            .iter()
                                            .zip(self.col_inds.iter())) {
            let i = i.index();
            let j = j.index();
            let start = indptr[i].index();
            let stop = start + row_counts[i].index();
            let col_exists = {
                let mut col_exists = false;
                let iter = indices[start..stop]
                               .iter()
                               .zip(data[start..stop].iter_mut());
                for (&col_cell, mut data_cell) in iter {
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
        csmat::raw::convert_storage(csmat::CompressedStorage::CSR,
                                    self.shape(),
                                    &indptr,
                                    &indices[..nnz],
                                    &data[..nnz],
                                    &mut out_indptr,
                                    &mut out_indices,
                                    &mut out_data);
        CsMatI {
            storage: csmat::CompressedStorage::CSC,
            nrows: self.rows,
            ncols: self.cols,
            indptr: out_indptr,
            indices: out_indices,
            data: out_data
        }
    }

    /// Create a CSR matrix from this triplet matrix
    pub fn to_csr(&self) -> CsMatI<N, I>
    where N: Clone + Num
    {
        let res = self.transpose_view().to_csc();
        res.transpose_into()
    }
}


impl<'a, N, I: SpIndex> TriMatViewMutBase<'a, N, I> {
    /// The number of rows of the matrix
    pub fn rows(&self) -> usize {
        self.borrowed().rows()
    }

    /// The number of cols of the matrix
    pub fn cols(&self) -> usize {
        self.borrowed().cols()
    }

    /// The shape of the matrix, as a `(rows, cols)` tuple
    pub fn shape(&self) -> (usize, usize) {
        self.borrowed().shape()
    }

    /// The number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.borrowed().nnz()
    }

    /// The non-zero row indices
    pub fn row_inds(&self) -> &[I] {
        self.borrowed().row_inds()
    }

    /// The non-zero column indices
    pub fn col_inds(&self) -> &[I] {
        self.borrowed().col_inds()
    }

    /// The non-zero values
    pub fn data(&self) -> &[N] {
        self.borrowed().data()
    }

    /// Return a view of this matrix
    pub fn borrowed(&self) -> TriMatViewBase<N, I> {
        TriMatViewBase {
            rows: self.rows,
            cols: self.cols,
            row_inds: &self.row_inds[..],
            col_inds: &self.col_inds[..],
            data: &self.data[..],
        }
    }

    /// Get a transposed view of this matrix
    pub fn transpose_view(&self) -> TriMatViewBase<N, I> {
        self.borrowed().transpose_view()
    }

    /// Replace a non-zero value at the given index.
    /// Indices can be obtained using find_locations.
    pub fn set_triplet(&mut self,
                       TripletIndex(triplet_ind): TripletIndex,
                       row: usize,
                       col: usize,
                       val: N) {
        self.row_inds[triplet_ind] = I::from_usize(row);
        self.col_inds[triplet_ind] = I::from_usize(col);
        self.data[triplet_ind] = val;
    }

    /// Create a CSC matrix from this triplet matrix
    pub fn to_csc(&self) -> CsMatI<N, I>
    where N: Clone + Num
    {
        self.borrowed().to_csc()
    }

    /// Create a CSR matrix from this triplet matrix
    pub fn to_csr(&self) -> CsMatI<N, I>
    where N: Clone + Num
    {
        self.borrowed().to_csr()
    }
}

#[cfg(test)]
mod test {

    use super::{TriMat, TriMatI};
    use sparse::{CsMat, CsMatI};

    #[test]
    fn triplet_incremental() {
        let mut triplet_mat = TriMatI::with_capacity((4, 4), 6);
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

        let csc: CsMatI<_, i32> = triplet_mat.to_csc();
        let expected = CsMatI::new_csc((4, 4),
                                            vec![0, 2, 3, 4, 6],
                                            vec![0, 1, 0, 3, 2, 3],
                                            vec![1., 3., 2., 5., 4., 6.]);
        assert_eq!(csc, expected);
    }

    #[test]
    fn triplet_unordered() {
        let mut triplet_mat = TriMat::with_capacity((4, 4), 6);
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
        let expected = CsMat::new_csc((4, 4),
                                           vec![0, 2, 3, 4, 6],
                                           vec![0, 1, 0, 3, 2, 3],
                                           vec![1., 3., 2., 5., 4., 6.]);
        assert_eq!(csc, expected);
    }

    #[test]
    fn triplet_additions() {
        let mut triplet_mat = TriMat::with_capacity((4, 4), 6);
        // |1 2    |
        // |3      |
        // |      4|
        // |    5 6|

        // here we test the additive properties of triples
        // the (3, 2) nnz element is specified twice
        triplet_mat.add_triplet(0, 1, 2.);
        triplet_mat.add_triplet(0, 0, 1.);
        triplet_mat.add_triplet(3, 2, 3.);
        triplet_mat.add_triplet(1, 0, 3.);
        triplet_mat.add_triplet(2, 3, 4.);
        triplet_mat.add_triplet(3, 3, 6.);
        triplet_mat.add_triplet(3, 2, 2.);

        let csc = triplet_mat.to_csc();
        let expected = CsMat::new_csc((4, 4),
                                           vec![0, 2, 3, 4, 6],
                                           vec![0, 1, 0, 3, 2, 3],
                                           vec![1., 3., 2., 5., 4., 6.]);
        assert_eq!(csc, expected);
    }

    #[test]
    fn triplet_from_vecs() {
        // |1 2    |
        // |3      |
        // |      4|
        // |    5 6|
        // |  7   8|
        let row_inds = vec![0, 0, 1, 2, 3, 3, 4, 4];
        let col_inds = vec![0, 1, 0, 3, 2, 3, 1, 3];
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let triplet_mat = super::TriMat::from_triplets((5, 4),
                                                           row_inds,
                                                           col_inds,
                                                           data);

        let csc = triplet_mat.to_csc();
        let expected = CsMat::new_csc((5, 4),
                                           vec![0, 2, 4, 5, 8],
                                           vec![0, 1, 0, 4, 3, 2, 3, 4],
                                           vec![1, 3, 2, 7, 5, 4, 6, 8]);

        assert_eq!(csc, expected);
    }

    #[test]
    fn triplet_mutate_entry() {
        let mut triplet_mat = TriMat::with_capacity((4, 4), 6);
        triplet_mat.add_triplet(0, 0, 1.);
        triplet_mat.add_triplet(0, 1, 2.);
        triplet_mat.add_triplet(1, 0, 3.);
        triplet_mat.add_triplet(2, 3, 4.);
        triplet_mat.add_triplet(3, 2, 5.);
        triplet_mat.add_triplet(3, 3, 6.);

        let locations = triplet_mat.find_locations(2, 3);
        assert_eq!(locations.len(), 1);
        triplet_mat.set_triplet(locations[0], 2, 3, 0.);


        let csc = triplet_mat.to_csc();
        let expected = CsMat::new_csc((4, 4),
                                           vec![0, 2, 3, 4, 6],
                                           vec![0, 1, 0, 3, 2, 3],
                                           vec![1., 3., 2., 5., 0., 6.]);
        assert_eq!(csc, expected);
    }

    #[test]
    fn triplet_to_csr() {
        let mut triplet_mat = TriMat::with_capacity((4, 4), 6);
        // |1 2    |
        // |3      |
        // |      4|
        // |    5 6|

        // here we test the additive properties of triples
        // the (3, 2) nnz element is specified twice
        triplet_mat.add_triplet(0, 1, 2.);
        triplet_mat.add_triplet(0, 0, 1.);
        triplet_mat.add_triplet(3, 2, 3.);
        triplet_mat.add_triplet(1, 0, 3.);
        triplet_mat.add_triplet(2, 3, 4.);
        triplet_mat.add_triplet(3, 3, 6.);
        triplet_mat.add_triplet(3, 2, 2.);

        let csr = triplet_mat.to_csr();
        let expected = CsMat::new_csc((4, 4),
                                           vec![0, 2, 3, 4, 6],
                                           vec![0, 1, 0, 3, 2, 3],
                                           vec![1., 3., 2., 5., 4., 6.])
                           .to_csr();
        assert_eq!(csr, expected);
    }

    #[test]
    fn triplet_complex() {
        // |1       6       2|
        // |1         1     2|
        // |1 2   3     3   2|
        // |1   9     4     2|
        // |1     5         2|
        // |1         7   8 2|
        let mut triplet_mat = TriMat::with_capacity((6, 9), 22);

        triplet_mat.add_triplet(5, 8, 1); // (a) push 1 later
        triplet_mat.add_triplet(0, 0, 1);
        triplet_mat.add_triplet(0, 8, 2);
        triplet_mat.add_triplet(0, 4, 2); // (b) push 4 later
        triplet_mat.add_triplet(2, 0, 1);
        triplet_mat.add_triplet(2, 1, 2);
        triplet_mat.add_triplet(2, 3, 2); // (c) push 1 later
        triplet_mat.add_triplet(2, 6, 3);
        triplet_mat.add_triplet(2, 8, 2);
        triplet_mat.add_triplet(1, 0, 1);
        triplet_mat.add_triplet(1, 5, 1);
        triplet_mat.add_triplet(1, 8, 1); // (d) push 1 later
        triplet_mat.add_triplet(0, 4, 4); // push the missing 4 (b)
        triplet_mat.add_triplet(3, 8, 2);
        triplet_mat.add_triplet(3, 5, 4);
        triplet_mat.add_triplet(5, 8, 1); // push the missing 1 (a)
        triplet_mat.add_triplet(3, 2, 9);
        triplet_mat.add_triplet(3, 0, 1);
        triplet_mat.add_triplet(4, 0, 1);
        triplet_mat.add_triplet(4, 8, 2);
        triplet_mat.add_triplet(1, 8, 1); // push the missing 1 (d)
        triplet_mat.add_triplet(4, 3, 5);
        triplet_mat.add_triplet(5, 0, 1);
        triplet_mat.add_triplet(5, 5, 7);
        triplet_mat.add_triplet(2, 3, 1); // push the missing 1 (c)
        triplet_mat.add_triplet(5, 7, 8);

        let csc = triplet_mat.to_csc();

        let expected = CsMat::new_csc((6, 9),
                                           vec![0, 6, 7, 8, 10, 11,
                                                14, 15, 16, 22],
                                           vec![0, 1, 2, 3, 4, 5, 2,
                                                3, 2, 4, 0, 1, 3, 5,
                                                2, 5, 0, 1, 2, 3, 4,
                                                5],
                                           vec![1, 1, 1, 1, 1, 1, 2,
                                                9, 3, 5, 6, 1, 4, 7,
                                                3, 8, 2, 2, 2, 2, 2,
                                                2]);

        assert_eq!(csc, expected);

        let csr = triplet_mat.to_csr();

        assert_eq!(csr, expected.to_csr());
    }
}
