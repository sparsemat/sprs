///! Simple structures for interoperability with dense matrices

use std::ops::{Deref, DerefMut};
use std::iter::Map;
use std::slice::{Chunks, ChunksMut};
use num::traits::Num;
use dense_mats::{DenseMatView, DenseMatViewMut, StorageOrder,
                 DenseMatOwned,};
use errors::SprsError;

/// A simple dense matrix
pub struct DMat<N, Storage>
where Storage: Deref<Target=[N]> {
    data: Storage,
    rows: usize,
    cols: usize,
    strides: [usize; 2],
}

pub type DMatView<'a, N> = DMat<N, &'a [N]>;
pub type DMatOwned<N> = DMat<N, Vec<N>>;

/// A simple dense vector
pub struct DVec<N, Storage>
where Storage: Deref<Target=[N]> {
    data: Storage,
    dim: usize,
    stride: usize,
}

pub type DVecView<'a, N> = DVec<N, &'a [N]>;
pub type DVecOwned<N> = DVec<N, Vec<N>>;

impl<N, Storage> DenseMatView<N> for DMat<N, Storage>
where Storage: Deref<Target=[N]> {

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn strides(&self) -> [usize; 2] {
        self.strides
    }

    fn data(&self) -> &[N] {
        &self.data[..]
    }
}

impl<N, Storage> DenseMatViewMut<N> for DMat<N, Storage>
where Storage: DerefMut<Target=[N]> {
    fn data_mut(&mut self) -> &mut [N] {
        &mut self.data[..]
    }
}

impl<N> DenseMatOwned<N> for DMat<N, Vec<N>> {
    fn into_data(self) -> Vec<N> {
        self.data
    }
}

impl<N, Storage> DMat<N, Storage>
where Storage: Deref<Target=[N]> {

    /// Create a view of a matrix implementing DenseMatView
    pub fn wrap_view<'a, Mat: 'a + DenseMatView<N>>(m: &'a Mat)
    -> DMatView<'a, N>
    where N: 'a {
        DMat {
            data: m.data(),
            rows: m.rows(),
            cols: m.cols(),
            strides: m.strides(),
        }
    }

    /// Create from a matrix implementing DenseMatOwned
    pub fn from_owned<Mat: DenseMatOwned<N>>(m: Mat) -> DMatOwned<N> {
        let rows = m.rows();
        let cols = m.cols();
        let strides = m.strides();
        DMat {
            data: m.into_data(),
            rows: rows,
            cols: cols,
            strides: strides,
        }
    }

    /// Create a dense matrix from owned data
    pub fn new_owned(data: Vec<N>, rows: usize,
                     cols: usize, strides: [usize;2]) -> DMatOwned<N> {
        DMat {
            data: data,
            rows: rows,
            cols: cols,
            strides: strides,
        }
    }

    /// Create an all-zero dense matrix
    pub fn zeros(rows: usize, cols: usize,
                 order: StorageOrder) -> DMatOwned<N>
    where N: Num + Copy {
        let strides = match order {
            StorageOrder::RowMaj => [cols, 1],
            StorageOrder::ColMaj => [1, rows],
        };
        DMat {
            data: vec![N::zero(); rows*cols],
            rows: rows,
            cols: cols,
            strides: strides,
        }
    }

    /// Get a view into the specified row
    pub fn row(&self, i: usize) -> Result<DVecView<N>, SprsError> {
        if i > self.rows {
            return Err(SprsError::OutOfBoundsIndex);
        }
        let start = i * self.strides[0];
        let stop = (i + 1) * self.strides[0];
        let data = &self.data[start..stop];
        Ok(DVec {
            data: data,
            dim: self.cols,
            stride: self.strides[1],
        })
    }

    /// Get a view into the specified column
    pub fn col(&self, j: usize) -> Result<DVecView<N>, SprsError> {
        if j > self.cols {
            return Err(SprsError::OutOfBoundsIndex);
        }
        let start = j * self.strides[1];
        let stop = (j + 1) * self.strides[1];
        let data = &self.data[start..stop];
        Ok(DVec {
            data: data,
            dim: self.cols,
            stride: self.strides[0],
        })
    }
}

fn take_first<N>(chunk: &[N]) -> &N {
    &chunk[0]
}

fn take_first_mut<N>(chunk: &mut [N]) -> &mut N {
    &mut chunk[0]
}


impl<N, Storage> DVec<N, Storage>
where Storage: Deref<Target=[N]> {

    /// Iterate over a dense vector's values by reference
    pub fn iter(&self) -> Map<Chunks<N>, fn(&[N]) -> &N> {
        self.data.chunks(self.stride).map(take_first)
    }
}

impl<N, Storage> DVec<N, Storage>
where Storage: DerefMut<Target=[N]> {

    /// Iterate over a dense vector's values by mutable reference
    pub fn iter_mut(&mut self) -> Map<ChunksMut<N>, fn(&mut [N]) -> &mut N> {
        self.data.chunks_mut(self.stride).map(take_first_mut)
    }
}

