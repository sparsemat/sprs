///! Sparse linear algebra
///!
///! This module contains solvers for sparse linear systems. Currently
///! there are solver for sparse triangular systems and symmetric systems.


use num::traits::Num;
use std::iter::IntoIterator;

#[cfg(feature = "lgpl")]
pub use self::cholesky::ldl_symbolic;

#[cfg(feature = "lgpl")]
pub mod cholesky;

pub mod trisolve;
pub mod etree;

/// Diagonal solve
pub fn diag_solve<'a, N, I1, I2>(diag: I1, x: I2)
where N: 'a + Copy + Num,
      I1: IntoIterator<Item = &'a N>,
      I2: IntoIterator<Item = &'a mut N>
{

    for (xv, dv) in x.into_iter().zip(diag.into_iter()) {
        *xv = *xv / *dv;
    }
}
