/// Dense vector utility functions

use num::traits::{Num, Zero};

pub fn zeros<N: Num> (n: usize) -> Vec<N> {
    return (0..n).map(|x| Zero::zero()).collect::<Vec<N>>();
}
