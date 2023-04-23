use super::{SuiteSparseInt, SuiteSparseLong};
use core::ffi::{c_double, c_void};

extern "C" {
    pub fn umfpack_di_symbolic(
        n_row: SuiteSparseInt,
        n_col: SuiteSparseInt,
        Ap: *const SuiteSparseInt,
        Ai: *const SuiteSparseInt,
        Ax: *const c_double,
        Symbolic: *mut *mut c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> SuiteSparseInt;

    pub fn umfpack_dl_symbolic(
        n_row: SuiteSparseLong,
        n_col: SuiteSparseLong,
        Ap: *const SuiteSparseLong,
        Ai: *const SuiteSparseLong,
        Ax: *const c_double,
        Symbolic: *mut *mut c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> SuiteSparseLong;
}
