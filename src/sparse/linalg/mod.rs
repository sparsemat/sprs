/// Sparse linear algebra


pub use self::cholesky::ldl_symbolic;
use num::traits::Num;
use std::iter::IntoIterator;

pub mod cholesky;
pub mod trisolve;

/// Diagonal solve
pub fn diag_solve<'a, N, I1, I2>(diag: I1, x: I2)
where N: 'a + Copy + Num,
      I1: IntoIterator<Item=&'a N>,
      I2: IntoIterator<Item=&'a mut N> {

    for (xv, dv) in x.into_iter().zip(diag.into_iter()) {
        *xv = *xv / *dv;
    }
}
