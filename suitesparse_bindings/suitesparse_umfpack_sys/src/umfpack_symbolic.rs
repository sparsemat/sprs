use libc::{c_double, c_int, c_void};

// C function signature for umfpack_symbolic
extern "C" {
    fn umfpack_symbolic(
        n_row: c_int,
        n_col: c_int,
        Ap: *const c_int,
        Ai: *const c_int,
        Ax: *const c_double,
        Symbolic: *mut *mut c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> c_int;
}

// Rust wrapper function for umfpack_symbolic
pub fn umfpack_symbolic_wrapper(
    n_row: c_int,               // Number of rows
    n_col: c_int,               // Number of columns
    ap: &[c_int],               // Column pointers for sparse matrix A
    ai: &[c_int],               // Row indices for sparse matrix A
    ax: &[c_double],            // Values for sparse matrix A
    symbolic: &mut *mut c_void, // Opaque representation of symbolic decomposition; not populated yet
    control: &[c_double], // Control parameters; null pointer -> default settings
    info: &mut [c_int],   // Info readout; null pointer -> ignore readout
) -> c_int {
    unsafe {
        umfpack_symbolic(
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
