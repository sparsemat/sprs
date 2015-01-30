/// Dense vector utility functions

use std::num::Int;

pub fn zeros<N: Int> (n: usize) -> Vec<N> {
    return (0..n).map(|x| Int::zero()).collect::<Vec<N>>();
}
