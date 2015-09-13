///! Simple structures for interoperability with dense matrices

use std::ops::{Deref, DerefMut};
use num::traits::Num;
use dense_mats::{DenseMatView, DenseMatViewMut, StorageOrder,
                 DenseMatOwned,};

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

}

