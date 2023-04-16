#[cfg(target_os = "windows")]
pub type SuiteSparseLong = libc::c_longlong;
#[cfg(not(target_os = "windows"))]
pub type SuiteSparseLong = libc::c_long;

pub type SuiteSparseInt = libc::c_int;

pub mod umfpack_free_numeric;
pub mod umfpack_free_symbolic;
pub mod umfpack_get_lunz;
pub mod umfpack_get_numeric;
pub mod umfpack_numeric;
pub mod umfpack_solve;
pub mod umfpack_symbolic;

pub use umfpack_free_numeric::{umfpack_di_free_numeric, umfpack_dl_free_numeric};
pub use umfpack_free_symbolic::{umfpack_di_free_symbolic, umfpack_dl_free_symbolic};
pub use umfpack_get_lunz::{umfpack_di_get_lunz, umfpack_dl_get_lunz};
pub use umfpack_numeric::{umfpack_di_numeric_wrapper, umfpack_dl_numeric_wrapper};
pub use umfpack_solve::{umfpack_di_solve_wrapper, umfpack_dl_solve_wrapper};
pub use umfpack_symbolic::{umfpack_di_symbolic_wrapper, umfpack_dl_symbolic_wrapper};
