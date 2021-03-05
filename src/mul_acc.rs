//! Multiply-accumulate (MAC) trait and implementations
//! It's useful to define our own MAC trait as it's the main primitive we use
//! in matrix products, and defining it ourselves means we can define an
//! implementation that does not require cloning, which should prove useful
//! when defining sparse matrices per blocks (eg BSR, BSC)

/// Trait for types that have a multiply-accumulate operation, as required
/// in dot products and matrix products.
///
/// This trait is automatically implemented for numeric types that are `Copy`,
/// however the implementation is open for more complex types, to allow them
/// to provide the most performant implementation. For instance, we could have
/// a default implementation for numeric types that are `Clone`, but it would
/// make possibly unnecessary copies.
pub trait MulAcc {
    /// Multiply and accumulate in this variable, formally `*self += a * b`.
    fn mul_acc(&mut self, a: &Self, b: &Self);
}

impl<N> MulAcc for N
where
    N: Copy + num_traits::MulAdd<Output = N>,
{
    fn mul_acc(&mut self, a: &Self, b: &Self) {
        *self = a.mul_add(*b, *self);
    }
}

#[cfg(test)]
mod tests {
    use super::MulAcc;

    #[test]
    fn mul_acc_f64() {
        let mut a = 1f64;
        let b = 2.;
        let c = 3.;
        a.mul_acc(&b, &c);
        assert_eq!(a, 7.);
    }
}
