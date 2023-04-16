use libc::c_void;

extern "C" {
    fn umfpack_di_free_symbolic(Symbolic: *mut *mut c_void);
    fn umfpack_dl_free_symbolic(Symbolic: *mut *mut c_void);
}

pub fn umfpack_di_free_symbolic_wrapper(symbolic: &mut *mut c_void) {
    unsafe {
        umfpack_di_free_symbolic(symbolic as *mut *mut c_void);
    }
}

pub fn umfpack_dl_free_symbolic_wrapper(symbolic: &mut *mut c_void) {
    unsafe {
        umfpack_dl_free_symbolic(symbolic as *mut *mut c_void);
    }
}
