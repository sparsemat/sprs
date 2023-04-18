use super::{SuiteSparseInt, SuiteSparseLong};
use core::ffi::{c_double, c_void};

extern "C" {
    pub fn umfpack_di_solve(
        sys: SuiteSparseInt,
        Ap: *const SuiteSparseInt,
        Ai: *const SuiteSparseInt,
        Ax: *const c_double,
        X: *mut c_double,
        B: *const c_double,
        Numeric: *const c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> SuiteSparseInt;

    pub fn umfpack_dl_solve(
        sys: SuiteSparseLong,
        Ap: *const SuiteSparseLong,
        Ai: *const SuiteSparseLong,
        Ax: *const c_double,
        X: *mut c_double,
        B: *const c_double,
        Numeric: *const c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> SuiteSparseLong;
}
