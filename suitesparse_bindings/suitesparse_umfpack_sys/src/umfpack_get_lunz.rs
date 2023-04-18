use super::{SuiteSparseInt, SuiteSparseLong};
use core::ffi::c_void;

extern "C" {
    pub fn umfpack_di_get_lunz(
        lnz: *mut SuiteSparseInt,
        unz: *mut SuiteSparseInt,
        nrow: *mut SuiteSparseInt,
        ncol: *mut SuiteSparseInt,
        nz_udiag: *mut SuiteSparseInt,
        numeric: *const c_void,
    ) -> SuiteSparseInt;

    pub fn umfpack_dl_get_lunz(
        lnz: *mut SuiteSparseLong,
        unz: *mut SuiteSparseLong,
        nrow: *mut SuiteSparseLong,
        ncol: *mut SuiteSparseLong,
        nz_udiag: *mut SuiteSparseLong,
        numeric: *const c_void,
    ) -> SuiteSparseLong;
}
