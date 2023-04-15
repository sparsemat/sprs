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
    ap: &[c_int],
    ai: &[c_int],
    ax: &[c_double],
    x: &mut [c_double],
    b: &[c_double],
    numeric: *const c_void,
    control: &[c_double],
    info: &mut [c_int],
) -> c_int {
    // let n = x.len() as c_int;
    // let m = b.len() as c_int;

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