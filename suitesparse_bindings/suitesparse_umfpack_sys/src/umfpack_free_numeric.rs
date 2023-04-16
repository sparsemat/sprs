use libc::{c_int, c_void};

// Define the C function signature for umfpack_free_numeric
extern "C" {
    fn umfpack_free_numeric(Numeric: *mut *mut c_void);
}

// Define a Rust wrapper function for umfpack_free_numeric
pub fn umfpack_free_numeric_wrapper(numeric: &mut *mut c_void) {
    unsafe {
        umfpack_free_numeric(numeric as *mut *mut c_void);
    }
}
