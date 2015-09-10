
pub use self::csmat::{CompressedStorage,
                      CsMat,
                      CsMatOwned,
                      CsMatView,
};

pub use self::vec::{CsVec,
                    CsVecOwned,
                    CsVecView,
};


pub mod csmat;
pub mod vec;
pub mod permutation;
pub mod prod;
pub mod binop;
pub mod construct;
pub mod linalg;
pub mod symmetric;
pub mod compressed;

