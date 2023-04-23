use core::ffi::c_void;

extern "C" {
    pub fn umfpack_di_free_numeric(Numeric: *mut *mut c_void);
    pub fn umfpack_dl_free_numeric(Numeric: *mut *mut c_void);
}
