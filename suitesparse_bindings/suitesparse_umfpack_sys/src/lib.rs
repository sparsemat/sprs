//! Raw C bindings to some (but not all) UMFPACK functions.

#[cfg(target_os = "windows")]
pub type SuiteSparseLong = core::ffi::c_longlong;
#[cfg(not(target_os = "windows"))]
pub type SuiteSparseLong = core::ffi::c_long;

pub type SuiteSparseInt = core::ffi::c_int;

pub mod umfpack_free_numeric;
pub mod umfpack_free_symbolic;
pub mod umfpack_get_lunz;
pub mod umfpack_get_numeric;
pub mod umfpack_numeric;
pub mod umfpack_solve;
pub mod umfpack_symbolic;

pub use umfpack_free_numeric::{
    umfpack_di_free_numeric, umfpack_dl_free_numeric,
};
pub use umfpack_free_symbolic::{
    umfpack_di_free_symbolic, umfpack_dl_free_symbolic,
};
pub use umfpack_get_lunz::{umfpack_di_get_lunz, umfpack_dl_get_lunz};
pub use umfpack_get_numeric::{umfpack_di_get_numeric, umfpack_dl_get_numeric};
pub use umfpack_numeric::{umfpack_di_numeric, umfpack_dl_numeric};
pub use umfpack_solve::{umfpack_di_solve, umfpack_dl_solve};
pub use umfpack_symbolic::{umfpack_di_symbolic, umfpack_dl_symbolic};
