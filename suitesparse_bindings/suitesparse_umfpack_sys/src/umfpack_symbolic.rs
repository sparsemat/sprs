use super::{SuiteSparseInt, SuiteSparseLong};
use libc::{c_double, c_void};

extern "C" {
    fn umfpack_di_symbolic(
        n_row: SuiteSparseInt,
        n_col: SuiteSparseInt,
        Ap: *const SuiteSparseInt,
        Ai: *const SuiteSparseInt,
        Ax: *const c_double,
        Symbolic: *mut *mut c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> SuiteSparseInt;

    fn umfpack_dl_symbolic(
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

pub fn umfpack_di_symbolic_wrapper(
    n_row: SuiteSparseInt,
    n_col: SuiteSparseInt,
    ap: &[SuiteSparseInt],
    ai: &[SuiteSparseInt],
    ax: &[c_double],
    symbolic: &mut *mut c_void,
    control: &[c_double],
    info: &mut [SuiteSparseInt],
) -> SuiteSparseInt {
    unsafe {
        umfpack_di_symbolic(
            n_row,
            n_col,
            ap.as_ptr(),
            ai.as_ptr(),
            ax.as_ptr(),
            symbolic as *mut *mut c_void,
            control.as_ptr(),
            info.as_mut_ptr() as *mut c_void,
        )
    }
}

pub fn umfpack_dl_symbolic_wrapper(
    n_row: SuiteSparseLong,
    n_col: SuiteSparseLong,
    ap: &[SuiteSparseLong],
    ai: &[SuiteSparseLong],
    ax: &[c_double],
    symbolic: &mut *mut c_void,
    control: &[c_double],
    info: &mut [SuiteSparseLong],
) -> SuiteSparseLong {
    unsafe {
        umfpack_dl_symbolic(
            n_row,
            n_col,
            ap.as_ptr(),
            ai.as_ptr(),
            ax.as_ptr(),
            symbolic as *mut *mut c_void,
            control.as_ptr(),
            info.as_mut_ptr() as *mut c_void,
        )
    }
}
