use crate::Ix1;
use ndarray::{self, ArrayBase};
use num_traits::identities::Zero;

/// A trait for types representing dense vectors, useful for expressing
/// algorithms such as sparse-dense dot product, or linear solves.
pub trait DenseVector {
    type Owned;
    type Scalar;

    /// The dimension of the vector
    fn dim(&self) -> usize;

    /// Random access to an element in the vector.
    ///
    /// # Panics
    ///
    /// If the index is out of bounds
    fn index(&self, idx: usize) -> &Self::Scalar;

    /// Create an owned version of this dense vector type, filled with zeros
    fn zeros(dim: usize) -> Self::Owned;

    /// Copies this vector into an owned version
    fn to_owned(&self) -> Self::Owned;
}

impl<N: Zero + Clone> DenseVector for [N] {
    type Owned = Vec<N>;
    type Scalar = N;

    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }

    fn zeros(dim: usize) -> Self::Owned {
        vec![N::zero(); dim]
    }

    fn to_owned(&self) -> Self::Owned {
        self.to_vec()
    }
}

impl<'a, N: 'a + Zero + Clone> DenseVector for &'a [N] {
    type Owned = Vec<N>;
    type Scalar = N;

    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }

    fn zeros(dim: usize) -> Self::Owned {
        vec![N::zero(); dim]
    }

    fn to_owned(&self) -> Self::Owned {
        self.to_vec()
    }
}

impl<'a, N: 'a + Zero + Clone> DenseVector for &'a mut [N] {
    type Owned = Vec<N>;
    type Scalar = N;

    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }

    fn zeros(dim: usize) -> Self::Owned {
        vec![N::zero(); dim]
    }

    fn to_owned(&self) -> Self::Owned {
        self.to_vec()
    }
}

impl<N: Zero + Clone> DenseVector for Vec<N> {
    type Owned = Vec<N>;
    type Scalar = N;

    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }

    fn zeros(dim: usize) -> Self::Owned {
        vec![N::zero(); dim]
    }

    fn to_owned(&self) -> Self::Owned {
        self.to_vec()
    }
}

impl<'a, N: 'a + Zero + Clone> DenseVector for &'a Vec<N> {
    type Owned = Vec<N>;
    type Scalar = N;

    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }

    fn zeros(dim: usize) -> Self::Owned {
        vec![N::zero(); dim]
    }

    fn to_owned(&self) -> Self::Owned {
        self.to_vec()
    }
}

impl<'a, N: 'a + Zero + Clone> DenseVector for &'a mut Vec<N> {
    type Owned = Vec<N>;
    type Scalar = N;

    fn dim(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[idx]
    }

    fn zeros(dim: usize) -> Self::Owned {
        vec![N::zero(); dim]
    }

    fn to_owned(&self) -> Self::Owned {
        self.to_vec()
    }
}

impl<N, S> DenseVector for ArrayBase<S, Ix1>
where
    S: ndarray::Data<Elem = N>,
    N: Zero + Clone,
{
    type Owned = ndarray::Array<N, Ix1>;
    type Scalar = N;

    fn dim(&self) -> usize {
        self.shape()[0]
    }

    #[inline(always)]
    fn index(&self, idx: usize) -> &N {
        &self[[idx]]
    }

    fn zeros(dim: usize) -> Self::Owned {
        ndarray::Array::zeros(dim)
    }

    fn to_owned(&self) -> Self::Owned {
        self.to_owned()
    }
}

pub trait DenseVectorMut: DenseVector {
    /// Random mutable access to an element in the vector.
    ///
    /// # Panics
    ///
    /// If the index is out of bounds
    fn index_mut(&mut self, idx: usize) -> &mut Self::Scalar;
}

impl<'a, N: 'a + Zero + Clone> DenseVectorMut for [N] {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut N {
        &mut self[idx]
    }
}

impl<'a, N: 'a + Zero + Clone> DenseVectorMut for &'a mut [N] {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut N {
        &mut self[idx]
    }
}

impl<N: Zero + Clone> DenseVectorMut for Vec<N> {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut N {
        &mut self[idx]
    }
}

impl<'a, N: 'a + Zero + Clone> DenseVectorMut for &'a mut Vec<N> {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut N {
        &mut self[idx]
    }
}

impl<N, S> DenseVectorMut for ArrayBase<S, Ix1>
where
    S: ndarray::DataMut<Elem = N>,
    N: Zero + Clone,
{
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut N {
        &mut self[[idx]]
    }
}
