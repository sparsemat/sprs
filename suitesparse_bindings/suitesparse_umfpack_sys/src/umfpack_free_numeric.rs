use libc::c_void;

extern "C" {
    fn umfpack_di_free_numeric(Numeric: *mut *mut c_void);
    fn umfpack_dl_free_numeric(Numeric: *mut *mut c_void);
}

pub fn umfpack_di_free_numeric_wrapper(numeric: &mut *mut c_void) {
    unsafe {
        umfpack_di_free_numeric(numeric as *mut *mut c_void);
    }
}

pub fn umfpack_dl_free_numeric_wrapper(numeric: &mut *mut c_void) {
    unsafe {
        umfpack_dl_free_numeric(numeric as *mut *mut c_void);
    }
}