
pub use self::csmat::{
    CompressedStorage,
    CsMat,
    CsMatVec,
    CsMatView
};


mod csmat;
pub mod vec;
pub mod permutation;
pub mod prod;
pub mod binop;
pub mod construct;
pub mod linalg;
pub mod symmetric;
pub mod compressed;

