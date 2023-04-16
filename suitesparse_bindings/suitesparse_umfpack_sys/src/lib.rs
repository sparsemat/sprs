#[cfg(target_os = "windows")]
pub type SuiteSparseLong = libc::c_longlong;
#[cfg(not(target_os = "windows"))]
pub type SuiteSparseLong = libc::c_long;

pub mod umfpack_symbolic;
pub mod umfpack_numeric;
pub mod umfpack_solve;
