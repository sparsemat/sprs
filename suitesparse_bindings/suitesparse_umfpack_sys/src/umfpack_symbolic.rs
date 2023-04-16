use libc::{c_double, c_void};
use super::SuiteSparseLong;

// Define a C function signature for umfpack_symbolic
extern "C" {
    fn umfpack_symbolic(
        n_row: SuiteSparseLong,
        n_col: SuiteSparseLong,
        Ap: *const SuiteSparseLong,
        Ai: *const SuiteSparseLong,
        Ax: *const c_double,
        Symbolic: *mut *mut c_void,
        Control: *const c_double,
        Info: *mut c_void,
    ) -> SuiteSparseLong;
}

// Define a Rust wrapper function for umfpack_symbolic
pub fn umfpack_symbolic_wrapper(
    n_row: SuiteSparseLong,               // Number of rows
    n_col: SuiteSparseLong,               // Number of columns
    ap: &[SuiteSparseLong],               // Column pointers for sparse matrix A
    ai: &[SuiteSparseLong],               // Row indices for sparse matrix A
    ax: &[c_double],            // Values for sparse matrix A
    symbolic: &mut *mut c_void, // Opaque representation of symbolic decomposition; not populated yet
    control: &[c_double], // Control parameters; null pointer -> default settings
    info: &mut [SuiteSparseLong],   // Info readout; null pointer -> ignore readout
) -> SuiteSparseLong {
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
