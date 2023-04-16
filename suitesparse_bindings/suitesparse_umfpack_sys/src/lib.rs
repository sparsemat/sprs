#[cfg(target_os = "windows")]
pub type SuiteSparseLong = libc::c_longlong;
#[cfg(not(target_os = "windows"))]
pub type SuiteSparseLong = libc::c_long;

pub type SuiteSparseInt = libc::c_int;

pub mod umfpack_symbolic;
pub mod umfpack_numeric;
pub mod umfpack_solve;
pub mod umfpack_free_numeric;
pub mod umfpack_free_symbolic;
pub mod umfpack_get_lunz;
pub mod umfpack_get_numeric;