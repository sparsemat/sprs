use libc::{c_int, c_void};

// Define the C function signature for umfpack_free_symbolic
extern "C" {
    fn umfpack_free_symbolic(Symbolic: *mut *mut c_void);
}

// Define a Rust wrapper function for umfpack_free_symbolic
pub fn umfpack_free_symbolic_wrapper(symbolic: &mut *mut c_void) {
    unsafe {
        umfpack_free_symbolic(symbolic as *mut *mut c_void);
    }
}
