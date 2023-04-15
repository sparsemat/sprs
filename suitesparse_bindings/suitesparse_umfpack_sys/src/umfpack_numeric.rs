use libc::{c_double, c_int, c_void};

// Define the C function signature for umfpack_numeric
extern "C" {
    fn umfpack_numeric(
        Ap: *const c_int,
        Ai: *const c_int,
        Ax: *const c_double,
        Symbolic: *const c_void,
        Numeric: *mut *mut c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> c_int;
}

// Define a Rust wrapper function for umfpack_numeric
pub fn umfpack_numeric_wrapper(
    ap: &[c_int],
    ai: &[c_int],
    ax: &[c_double],
    symbolic: *const c_void,
    numeric: &mut *mut c_void,
    control: &[c_double],
    info: &mut [c_int],
) -> c_int {
    // let n = ap.len() - 1;

    unsafe {
        umfpack_numeric(
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
