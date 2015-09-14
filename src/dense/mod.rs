///! Simple structures for interoperability with dense matrices

use std::ops::{Deref, DerefMut, Range};
use std::iter::Map;
use std::slice::{Chunks, ChunksMut};
use num::traits::Num;
use dense_mats::{DenseMatView, DenseMatViewMut, StorageOrder,
                 DenseMatOwned,};
use errors::SprsError;

/// A simple dense matrix
#[derive(PartialEq, Debug)]
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
#[derive(PartialEq, Debug)]
pub struct DVec<N, Storage>
where Storage: Deref<Target=[N]> {
    data: Storage,
    dim: usize,
    stride: usize,
}

pub type DVecView<'a, N> = DVec<N, &'a [N]>;
pub type DVecViewMut<'a, N> = DVec<N, &'a mut [N]>;
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

impl<N> DMat<N, Vec<N>> {
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
}

impl<'a, N> DMat<N, &'a [N]> {

    /// Create a view of a matrix implementing DenseMatView
    pub fn wrap_view<Mat: 'a + DenseMatView<N>>(m: &'a Mat)
    -> DMatView<'a, N>
    where N: 'a {
        DMat {
            data: m.data(),
            rows: m.rows(),
            cols: m.cols(),
            strides: m.strides(),
        }
    }
}

impl<N, Storage> DMat<N, Storage>
where Storage: Deref<Target=[N]> {

    fn row_range_rowmaj(&self, i: usize) -> Range<usize> {
        let start = self.data_index(i, 0);
        let stop = self.data_index(i + 1, 0);
        start..stop
    }

    fn row_range_colmaj(&self, i: usize) -> Range<usize> {
        let start = self.data_index(i, 0);
        let stop = self.data_index(i + 1, self.cols() - 1);
        start..stop
    }

    fn col_range_rowmaj(&self, j: usize) -> Range<usize> {
        let start = self.data_index(0, j);
        let stop = self.data_index(self.rows() - 1, j + 1);
        start..stop
    }

    fn col_range_colmaj(&self, j: usize) -> Range<usize> {
        let start = self.data_index(0, j);
        let stop = self.data_index(0, j + 1);
        start..stop
    }



    /// Get a view into the specified row
    pub fn row(&self, i: usize) -> Result<DVecView<N>, SprsError> {
        if i >= self.rows {
            return Err(SprsError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::RowMaj => self.row_range_rowmaj(i),
            StorageOrder::ColMaj => self.row_range_colmaj(i),
        };
        Ok(DVec {
            data: &self.data[range],
            dim: self.cols,
            stride: self.strides[1],
        })
    }

    /// Get a view into the specified column
    pub fn col(&self, j: usize) -> Result<DVecView<N>, SprsError> {
        if j >= self.cols {
            return Err(SprsError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::RowMaj => self.col_range_rowmaj(j),
            StorageOrder::ColMaj => self.col_range_colmaj(j),
        };
        Ok(DVec {
            data: &self.data[range],
            dim: self.cols,
            stride: self.strides[0],
        })
    }
}

impl<N, Storage> DMat<N, Storage>
where Storage: DerefMut<Target=[N]> {
    /// Get a mutable view into the specified row
    pub fn row_mut(&mut self, i: usize) -> Result<DVecViewMut<N>, SprsError> {
        if i >= self.rows {
            return Err(SprsError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::RowMaj => self.row_range_rowmaj(i),
            StorageOrder::ColMaj => self.row_range_colmaj(i),
        };
        Ok(DVec {
            data: &mut self.data[range],
            dim: self.cols,
            stride: self.strides[1],
        })
    }

    /// Get a mutable view into the specified column
    pub fn col_mut(&mut self, j: usize) -> Result<DVecViewMut<N>, SprsError> {
        if j >= self.cols {
            return Err(SprsError::OutOfBoundsIndex);
        }
        let range = match self.ordering() {
            StorageOrder::RowMaj => self.col_range_rowmaj(j),
            StorageOrder::ColMaj => self.col_range_colmaj(j),
        };
        Ok(DVec {
            data: &mut self.data[range],
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

    /// The underlying data
    pub fn data(&self) -> &[N] {
        &self.data[..]
    }

    /// The number of dimensions
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The stride of this vector
    pub fn stride(&self) -> usize {
        self.stride
    }
}

impl<N, Storage> DVec<N, Storage>
where Storage: DerefMut<Target=[N]> {

    /// Iterate over a dense vector's values by mutable reference
    pub fn iter_mut(&mut self) -> Map<ChunksMut<N>, fn(&mut [N]) -> &mut N> {
        self.data.chunks_mut(self.stride).map(take_first_mut)
    }

    /// The underlying data as a mutable slice
    pub fn data_mut(&mut self) -> &mut [N] {
        &mut self.data[..]
    }
}

#[cfg(test)]
mod tests {

    use super::{DMat};
    use errors::SprsError;

    #[test]
    fn row_view() {

        let mat = DMat::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
                                  3, 3, [3, 1]);
        let view = mat.row(0).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 1);
        assert_eq!(view.data(), &[1., 1., 0.]);
        let view = mat.row(1).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 1);
        assert_eq!(view.data(), &[0., 1., 0.]);
        let view = mat.row(2).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 1);
        assert_eq!(view.data(), &[0., 0., 1.]);
        let res = mat.row(3);
        assert_eq!(res, Err(SprsError::OutOfBoundsIndex));
    }

    #[test]
    fn col_view() {

        let mat = DMat::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
                                  3, 3, [3, 1]);
        let view = mat.col(0).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 3);
        assert_eq!(view.data(), &[1., 1., 0., 0., 1., 0., 0.]);
        let view = mat.col(1).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 3);
        assert_eq!(view.data(), &[1., 0., 0., 1., 0., 0., 0.]);
        let view = mat.col(2).unwrap();
        assert_eq!(view.dim(), 3);
        assert_eq!(view.stride(), 3);
        assert_eq!(view.data(), &[0., 0., 1., 0., 0., 0., 1.]);
        let res = mat.col(3);
        assert_eq!(res, Err(SprsError::OutOfBoundsIndex));
    }

    #[test]
    fn row_iter() {
        let mat = DMat::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
                                  3, 3, [1, 3]);

        {
            let row = mat.row(0).unwrap();
            let mut iter = row.iter();
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), None);

            let row = mat.row(1).unwrap();
            let mut iter = row.iter();
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), None);

            let row = mat.row(2).unwrap();
            let mut iter = row.iter();
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), None);
        }

        let mut mat = mat;
        let mut row = mat.row_mut(0).unwrap();
        {
            let mut iter = row.iter_mut();
            *iter.next().unwrap() = 2.;
        }
        let mut iter = row.iter();
        assert_eq!(iter.next(), Some(&2.));
        assert_eq!(iter.next(), Some(&0.));
        assert_eq!(iter.next(), Some(&0.));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn col_iter() {
        let mat = DMat::new_owned(vec![1., 1., 0., 0., 1., 0., 0., 0., 1.],
                                  3, 3, [1, 3]);

        {
            let col = mat.col(0).unwrap();
            let mut iter = col.iter();
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), None);

            let col = mat.col(1).unwrap();
            let mut iter = col.iter();
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), None);

            let col = mat.col(2).unwrap();
            let mut iter = col.iter();
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&0.));
            assert_eq!(iter.next(), Some(&1.));
            assert_eq!(iter.next(), None);
        }

        let mut mat = mat;
        let mut col = mat.col_mut(0).unwrap();
        {
            let mut iter = col.iter_mut();
            *iter.next().unwrap() = 2.;
        }
        let mut iter = col.iter();
        assert_eq!(iter.next(), Some(&2.));
        assert_eq!(iter.next(), Some(&1.));
        assert_eq!(iter.next(), Some(&0.));
        assert_eq!(iter.next(), None);
    }
}
