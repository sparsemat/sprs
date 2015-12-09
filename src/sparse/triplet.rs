///! Triplet format matrix
///! Useful for building a matrix, but not for computations

/// Triplet matrix
pub struct TripletMat<N> {
    rows: usize,
    cols: usize,
    row_inds: Vec<usize>,
    col_inds: Vec<usize>,
    data: Vec<N>,
}

impl<N> TripletMat<N> {

    pub fn new(rows: usize, cols: usize) -> TripletMat<N> {
        TripletMat {
            rows: rows,
            cols: cols,
            row_inds: Vec::new(),
            col_inds: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn with_capacity(rows: usize,
                         cols: usize,
                         cap: usize)
                         -> TripletMat<N> {
        TripletMat {
            rows: rows,
            cols: cols,
            row_inds: Vec::with_capacity(cap),
            col_inds: Vec::with_capacity(cap),
            data: Vec::with_capacity(cap),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn add_triplet(&mut self, row: usize, col: usize, val: N) {
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
}
