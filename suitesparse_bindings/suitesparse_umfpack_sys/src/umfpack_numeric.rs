use super::{SuiteSparseInt, SuiteSparseLong};
use libc::{c_double, c_void};

extern "C" {
    fn umfpack_di_numeric(
        Ap: *const SuiteSparseInt,
        Ai: *const SuiteSparseInt,
        Ax: *const c_double,
        Symbolic: *const c_void,
        Numeric: *mut *mut c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> SuiteSparseInt;

    fn umfpack_dl_numeric(
        Ap: *const SuiteSparseLong,
        Ai: *const SuiteSparseLong,
        Ax: *const c_double,
        Symbolic: *const c_void,
        Numeric: *mut *mut c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> SuiteSparseLong;
}

pub fn umfpack_di_numeric_wrapper(
    ap: &[SuiteSparseInt],
    ai: &[SuiteSparseInt],
    ax: &[c_double],
    symbolic: *mut c_void,
    numeric: &mut *mut c_void,
    control: &[c_double],
    info: &mut [SuiteSparseInt],
) -> SuiteSparseInt {
    unsafe {
        umfpack_di_numeric(
            ap.as_ptr(),
            ai.as_ptr(),
            ax.as_ptr(),
            symbolic,
            numeric as *mut *mut c_void,
            control.as_ptr(),
            info.as_mut_ptr() as *mut c_void,
        )
    }
}

pub fn umfpack_dl_numeric_wrapper(
    ap: &[SuiteSparseLong],
    ai: &[SuiteSparseLong],
    ax: &[c_double],
    symbolic: *mut c_void,
    numeric: &mut *mut c_void,
    control: &[c_double],
    info: &mut [SuiteSparseLong],
) -> SuiteSparseLong {
    unsafe {
        umfpack_dl_numeric(
            ap.as_ptr(),
            ai.as_ptr(),
            ax.as_ptr(),
            symbolic,
            numeric as *mut *mut c_void,
            control.as_ptr(),
            info.as_mut_ptr() as *mut c_void,
        )
    }
}
