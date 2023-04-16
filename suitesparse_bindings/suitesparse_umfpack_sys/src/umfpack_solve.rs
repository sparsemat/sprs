use super::{SuiteSparseInt, SuiteSparseLong};
use libc::{c_double, c_void};

extern "C" {
    fn umfpack_di_solve(
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

    fn umfpack_dl_solve(
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

pub fn umfpack_di_solve_wrapper(
    sys: SuiteSparseInt,
    ap: &[SuiteSparseInt],
    ai: &[SuiteSparseInt],
    ax: &[c_double],
    x: &mut [c_double],
    b: &[c_double],
    numeric: *const c_void,
    control: &[c_double],
    info: &mut [SuiteSparseInt],
) -> SuiteSparseInt {
    unsafe {
        umfpack_di_solve(
            sys,
            ap.as_ptr(),
            ai.as_ptr(),
            ax.as_ptr(),
            x.as_mut_ptr(),
            b.as_ptr(),
            numeric,
            control.as_ptr(),
            info.as_mut_ptr() as *mut c_void,
        )
    }
}

pub fn umfpack_dl_solve_wrapper(
    sys: SuiteSparseLong,
    ap: &[SuiteSparseLong],
    ai: &[SuiteSparseLong],
    ax: &[c_double],
    x: &mut [c_double],
    b: &[c_double],
    numeric: *const c_void,
    control: &[c_double],
    info: &mut [SuiteSparseLong],
) -> SuiteSparseLong {
    unsafe {
        umfpack_dl_solve(
            sys,
            ap.as_ptr(),
            ai.as_ptr(),
            ax.as_ptr(),
            x.as_mut_ptr(),
            b.as_ptr(),
            numeric,
            control.as_ptr(),
            info.as_mut_ptr() as *mut c_void,
        )
    }
}
