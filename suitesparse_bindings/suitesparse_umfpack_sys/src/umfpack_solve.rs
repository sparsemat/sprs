use libc::{c_double, c_int, c_void};

// Define the C function signature for umfpack_solve
extern "C" {
    fn umfpack_solve(
        sys: c_int,
        Ap: *const c_int,
        Ai: *const c_int,
        Ax: *const c_double,
        X: *mut c_double,
        B: *const c_double,
        Numeric: *const c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> c_int;
}

// Define a Rust wrapper function for umfpack_solve
pub fn umfpack_solve_wrapper(
    sys: c_int,
    ap: &[c_int],           // Column pointers for sparse matrix A
    ai: &[c_int],           // Row indices for sparse matrix A
    ax: &[c_double],        // Values for sparse matrix A
    x: &mut [c_double],     // Values for Ax=b system x
    b: &[c_double], // Values for Ax=b system b, to be populated by solve
    numeric: *const c_void, // Opaque representation of LU decomposition; populated by umfpack_numeric
    control: &[c_double], // Control parameters; null pointer -> default settings
    info: &mut [c_int],   // Info readout; null pointer -> ignore readout
) -> c_int {
    unsafe {
        umfpack_solve(
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
