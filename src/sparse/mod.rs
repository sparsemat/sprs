
pub use self::csmat::{
    CompressedStorage,
    BorrowedCsMat,
    CsMat,
    check_csmat_structure,
};

// TODO: don't export low-level specialized methods
pub use self::prod::{
    mul_acc_mat_vec_csc
};

mod csmat;
mod prod;

