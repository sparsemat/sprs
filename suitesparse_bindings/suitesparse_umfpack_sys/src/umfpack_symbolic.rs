use libc::{c_double, c_int, c_void};

// Define the C function signature for umfpack_symbolic
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

// Define a Rust wrapper function for umfpack_symbolic
pub fn umfpack_symbolic_wrapper(
    n_row: c_int,
    n_col: c_int,
    ap: &[c_int],
    ai: &[c_int],
    ax: &[c_double],
    symbolic: &mut *mut c_void,
    control: &[c_double],
    info: &mut [c_int],
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
