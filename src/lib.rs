/*!
sprs
====

sprs is a sparse linear algebra library for Rust.

*/

extern crate num;

pub mod sparse;
pub mod errors;

pub use sparse::{CsMatVec, CsMatView};
pub use sparse::construct::{vstack, hstack, bmat};

#[cfg(test)]
mod test_data;
