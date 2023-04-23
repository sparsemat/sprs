use super::{SuiteSparseInt, SuiteSparseLong};
use core::ffi::{c_double, c_void};

extern "C" {
    pub fn umfpack_di_get_numeric(
        Lp: *mut SuiteSparseInt,
        Lj: *mut SuiteSparseInt,
        Lx: *mut c_double,
        Up: *mut SuiteSparseInt,
        Ui: *mut SuiteSparseInt,
        Ux: *mut c_double,
        P: *mut SuiteSparseInt,
        Q: *mut SuiteSparseInt,
        Dx: *mut c_double,
        do_recip: *const SuiteSparseInt,
        Rs: *mut c_double,
        Numeric: *const c_void,
    ) -> SuiteSparseInt;

    pub fn umfpack_dl_get_numeric(
        Lp: *mut SuiteSparseLong,
        Lj: *mut SuiteSparseLong,
        Lx: *mut c_double,
        Up: *mut SuiteSparseLong,
        Ui: *mut SuiteSparseLong,
        Ux: *mut c_double,
        P: *mut SuiteSparseLong,
        Q: *mut SuiteSparseLong,
        Dx: *mut c_double,
        do_recip: *const SuiteSparseLong,
        Rs: *mut c_double,
        Numeric: *const c_void,
    ) -> SuiteSparseLong;
}
