
pub use self::csmat::{
    CompressedStorage,
    CsMat,
};

// TODO: don't export low-level specialized methods
pub use self::prod::{
    mul_acc_mat_vec_csc,
    mul_acc_mat_vec_csr,
};

mod csmat;
pub mod vec;
pub mod permutation;
mod prod;
pub mod linalg;
pub mod symmetric;

