use libc::{c_double, c_void};
use super::SuiteSparseLong;

// Define a C function signature for umfpack_numeric
extern "C" {
    fn umfpack_numeric(
        Ap: *const SuiteSparseLong,
        Ai: *const SuiteSparseLong,
        Ax: *const c_double,
        Symbolic: *const c_void,
        Numeric: *mut *mut c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> SuiteSparseLong;
}

// Define a Rust wrapper function for umfpack_numeric
pub fn umfpack_numeric_wrapper(
    ap: &[SuiteSparseLong],               // Column pointers for sparse matrix A
    ai: &[SuiteSparseLong],               // Row indices for sparse matrix A
    ax: &[c_double],            // Values for sparse matrix A
    symbolic: *mut c_void, // Opaque representation of symbolic decomposition; populated by umfpack_symbolic
    numeric: &mut *mut c_void, // Opaque representation of LU decomposition; not populated yet
    control: &[c_double], // Control parameters; null pointer -> default settings
    info: &mut [SuiteSparseLong],   // Info readout; null pointer -> ignore readout
) -> SuiteSparseLong {
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
