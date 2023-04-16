use libc::{c_double, c_void};
use super::{SuiteSparseLong, SuiteSparseInt};

// Define the C function signature for umfpack_solve
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
}

// Define the C function signature for umfpack_solve
extern "C" {
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

// Define a Rust wrapper function for umfpack_solve
pub fn umfpack_di_solve_wrapper(
    sys: SuiteSparseInt,
    ap: &[SuiteSparseInt],           // Column pointers for sparse matrix A
    ai: &[SuiteSparseInt],           // Row indices for sparse matrix A
    ax: &[c_double],        // Values for sparse matrix A
    x: &mut [c_double],     // Values for Ax=b system x
    b: &[c_double], // Values for Ax=b system b, to be populated by solve
    numeric: *const c_void, // Opaque representation of LU decomposition; populated by umfpack_numeric
    control: &[c_double], // Control parameters; null pointer -> default settings
    info: &mut [SuiteSparseInt],   // Info readout; null pointer -> ignore readout
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

// Define a Rust wrapper function for umfpack_solve
pub fn umfpack_dl_solve_wrapper(
    sys: SuiteSparseLong,
    ap: &[SuiteSparseLong],           // Column pointers for sparse matrix A
    ai: &[SuiteSparseLong],           // Row indices for sparse matrix A
    ax: &[c_double],        // Values for sparse matrix A
    x: &mut [c_double],     // Values for Ax=b system x
    b: &[c_double], // Values for Ax=b system b, to be populated by solve
    numeric: *const c_void, // Opaque representation of LU decomposition; populated by umfpack_numeric
    control: &[c_double], // Control parameters; null pointer -> default settings
    info: &mut [SuiteSparseLong],   // Info readout; null pointer -> ignore readout
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
