/// Sparse linear algebra


pub use self::cholesky::ldl_symbolic;
use num::traits::Num;
use std::ops::{Add, Sub, Mul, Div};
use std::iter::IntoIterator;

pub mod cholesky;
pub mod trisolve;
pub mod etree;

/// Diagonal solve
pub fn diag_solve<'a, N, M, I1, I2>(diag: I1, x: I2)
where N: 'a + Copy + Num,
      M: 'a + Copy + Add<Output=M> + Sub<Output=M> + Mul<N, Output=M> + Div<N, Output=M>,
      I1: IntoIterator<Item = &'a N>,
      I2: IntoIterator<Item = &'a mut M>
{

    for (xv, dv) in x.into_iter().zip(diag.into_iter()) {
        *xv = *xv / *dv;
    }
}
