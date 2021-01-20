use crate::Ix1;
use ndarray::{self, ArrayBase};

/// A trait for types representing dense vectors, useful for expressing
/// algorithms such as sparse-dense dot product, or linear solves.
pub trait DenseVector<N> {
    /// The dimension of the vector
    fn dim(&self) -> usize;

    /// Random access to an element in the vector.
    ///
    /// # Panics
    ///
    /// If the index is out of bounds
    fn index(&self, idx: usize) -> &N;
}

impl<'a, N: 'a> DenseVector<N> for &'a [N] {
    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }
}

impl<'a, N: 'a> DenseVector<N> for &'a mut [N] {
    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }
}

impl<N> DenseVector<N> for Vec<N> {
    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }
}

impl<'a, N: 'a> DenseVector<N> for &'a Vec<N> {
    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }
}

impl<'a, N: 'a> DenseVector<N> for &'a mut Vec<N> {
    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }
}

impl<N, S> DenseVector<N> for ArrayBase<S, Ix1>
where
    S: ndarray::Data<Elem = N>,
{
    fn dim(&self) -> usize {
        self.shape()[0]
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[[idx]]
    }
}

pub trait DenseVectorMut<N>: DenseVector<N> {
    /// Random mutable access to an element in the vector.
    ///
    /// # Panics
    ///
    /// If the index is out of bounds
    fn index_mut(&mut self, idx: usize) -> &mut N;
}

impl<'a, N: 'a> DenseVectorMut<N> for &'a mut [N] {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut N {
        &mut self[idx]
    }
}

impl<N> DenseVectorMut<N> for Vec<N> {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut N {
        &mut self[idx]
    }
}

impl<'a, N: 'a> DenseVectorMut<N> for &'a mut Vec<N> {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut N {
        &mut self[idx]
    }
}

impl<N, S> DenseVectorMut<N> for ArrayBase<S, Ix1>
where
    S: ndarray::DataMut<Elem = N>,
{
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut N {
        &mut self[[idx]]
    }
}
