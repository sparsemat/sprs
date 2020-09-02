#[cfg(target_os = "windows")]
pub type SuiteSparseLong = libc::c_longlong;
#[cfg(not(target_os = "windows"))]
pub type SuiteSparseLong = libc::c_long;

pub type SuiteSparseInt = libc::c_int;

#[link(name = "camd")]
extern "C" {
    /// Find a permutation matrix P, represented by the permutation indices
    /// `p`, which reduces the fill-in of the symmetric sparse matrix A
    /// (represented by its index pointer `ap` and indices pointer `ai`)
    /// in Cholesky factorization (ie, the number of nonzeros of the Cholesky
    /// factorization of P A P^T is less than for the Cholesky factorization
    /// of A).
    ///
    /// If A is not symmetric, the ordering will be computed for A + A^T
    ///
    /// # Safety and constraints
    /// - `n` is the number of rows and columns of the matrix A and should be
    ///   positive
    /// - `ap` must be an array of length `n + 1`.
    /// - `ai` must be an array of length `ap[n]`. The performance is better
    ///   if `ai[ap[i]..ap[i+1]]` is always sorted and without duplicates.
    /// - `p` must be an array of length `n`. Its contents can be uninitialized.
    /// - `control` must be an array of size `CAMD_CONTROL`.
    /// - `info` must be an array of size `CAMD_INFO`.
    /// - `constraint` must be either the null pointer, or an array of size `n`.
    pub fn camd_order(
        n: SuiteSparseInt,
        ap: *const SuiteSparseInt,
        ai: *const SuiteSparseInt,
        p: *mut SuiteSparseInt,
        control: *mut libc::c_double,
        info: *mut libc::c_double,
        constraint: *mut SuiteSparseInt,
    ) -> SuiteSparseInt;

    /// Long version of `camd_order`, see its documentation.
    pub fn camd_l_order(
        n: SuiteSparseLong,
        ap: *const SuiteSparseLong,
        ai: *const SuiteSparseLong,
        p: *mut SuiteSparseLong,
        control: *mut libc::c_double,
        info: *mut libc::c_double,
        constraint: *mut SuiteSparseLong,
    ) -> SuiteSparseLong;

    /// Checks if the matrix A represented by its index pointer array
    /// `ap` and its indices array `ai` is suitable to pass to `camd_order`.
    ///
    /// Will return `CAMD_OK` if the matrix is suitable, and `CAMD_OK_BUT_JUMBLED`
    /// if the matrix has unsorted or duplicate row indices in one or more columns.
    /// The matrix must be square, ie `n_rows == n_cols`.
    ///
    /// Otherwise `CAMD_INVALID` will be returned.
    pub fn camd_valid(
        n_rows: SuiteSparseInt,
        n_cols: SuiteSparseInt,
        ap: *const SuiteSparseInt,
        ai: *const SuiteSparseInt,
    ) -> SuiteSparseInt;

    /// Long version of `camd_valid`, see its documentation.
    pub fn camd_l_valid(
        n_rows: SuiteSparseLong,
        n_cols: SuiteSparseLong,
        ap: *const SuiteSparseLong,
        ai: *const SuiteSparseLong,
    ) -> SuiteSparseLong;

    /// Check if the array `constraint`, of size `n`, is valid as input
    /// to `camd_order`. Returns `1` if valid, `0` otherwise.
    pub fn camd_cvalid(n: SuiteSparseInt, constraint: *const SuiteSparseInt);

    /// Long version of `camd_cvalid`, see its documentation.
    pub fn camd_l_cvalid(
        n: SuiteSparseLong,
        constraint: *const SuiteSparseLong,
    );

    /// Fill the `control` array of size `CAMD_CONTROL` with default values
    pub fn camd_defaults(control: *mut libc::c_double);

    /// Fill the `control` array of size `CAMD_CONTROL` with default values
    pub fn camd_l_defaults(control: *mut libc::c_double);

    /// Pretty print the `control` array of size `CAMD_CONTROL`
    pub fn camd_control(control: *const libc::c_double);

    /// Pretty print the `control` array of size `CAMD_CONTROL`
    pub fn camd_l_control(control: *const libc::c_double);

    /// Pretty print the `info` array of size `CAMD_INFO`
    pub fn camd_info(info: *const libc::c_double);

    /// Pretty print the `info` array of size `CAMD_INFO`
    pub fn camd_l_info(info: *const libc::c_double);
}

pub const CAMD_CONTROL: usize = 5;
pub const CAMD_INFO: usize = 20;

pub const CAMD_DENSE: usize = 0;
pub const CAMD_AGGRESSIVE: usize = 1;

pub const CAMD_STATUS: usize = 0;
pub const CAMD_N: usize = 1;
pub const CAMD_NZ: usize = 2;
pub const CAMD_SYMMETRY: usize = 3;
pub const CAMD_NZDIAG: usize = 4;
pub const CAMD_NZ_A_PLUS_AT: usize = 5;
pub const CAMD_NDENSE: usize = 6;
pub const CAMD_MEMORY: usize = 7;
pub const CAMD_NCMPA: usize = 8;
pub const CAMD_LNZ: usize = 9;
pub const CAMD_NDIV: usize = 10;
pub const CAMD_NMULTSUBS_LDL: usize = 11;
pub const CAMD_NMULTSUBS_LU: usize = 12;
pub const CAMD_DMAX: usize = 13;

pub const CAMD_OK: isize = 0;
pub const CAMD_OUT_OF_MEMORY: isize = -1;
pub const CAMD_INVALID: isize = -2;
pub const CAMD_OK_BUT_JUMBLED: isize = 1;

#[cfg(test)]
mod tests {
    use super::SuiteSparseInt;
    #[test]
    fn camd_valid() {
        // | 0 1 3 |
        // | 1 1 0 |
        // | 3 0 0 |
        let n: SuiteSparseInt = 3;
        let ap = &[0, 2, 4, 5];
        let ai = &[1, 2, 0, 1, 0];
        let valid =
            unsafe { super::camd_valid(n, n, ap.as_ptr(), ai.as_ptr()) };
        assert_eq!(valid, super::CAMD_OK as SuiteSparseInt);
        let ai = &[2, 1, 0, 1, 0];
        let valid =
            unsafe { super::camd_valid(n, n, ap.as_ptr(), ai.as_ptr()) };
        assert_eq!(valid, super::CAMD_OK_BUT_JUMBLED as SuiteSparseInt);
        let ai = &[1, 2, 0, 1, 1];
        let valid =
            unsafe { super::camd_valid(n, n, ap.as_ptr(), ai.as_ptr()) };
        assert_eq!(valid, super::CAMD_OK as SuiteSparseInt);
        let valid =
            unsafe { super::camd_valid(n, n + 1, ap.as_ptr(), ai.as_ptr()) };
        assert_eq!(valid, super::CAMD_INVALID as SuiteSparseInt);
        let valid = unsafe {
            super::camd_valid(n + 1, n + 1, ap.as_ptr(), ai.as_ptr())
        };
        assert_eq!(valid, super::CAMD_INVALID as SuiteSparseInt);

        // long version test, only test once as we now have the behavior tested
        // (we only want tot test the binding is correct now).
        use super::SuiteSparseLong as Long;
        let n: Long = 3;
        let ap: &[Long] = &[0, 2, 4, 5];
        let ai: &[Long] = &[1, 2, 0, 1, 0];
        let valid =
            unsafe { super::camd_l_valid(n, n, ap.as_ptr(), ai.as_ptr()) };
        assert_eq!(valid, super::CAMD_OK as Long);
    }

    #[test]
    fn camd_order() {
        // | 0 1 3 |
        // | 1 1 0 |
        // | 3 0 0 |
        let n: SuiteSparseInt = 3;
        let ap = &[0, 2, 4, 5];
        let ai = &[1, 2, 0, 1, 0];
        let mut perm = [0; 3];
        let mut control = [0.; super::CAMD_CONTROL];
        let mut info = [0.; super::CAMD_INFO];
        let constraint: *const SuiteSparseInt = std::ptr::null();
        unsafe {
            super::camd_defaults(control.as_mut_ptr());
        }
        let res = unsafe {
            super::camd_order(
                n,
                ap.as_ptr(),
                ai.as_ptr(),
                perm.as_mut_ptr(),
                control.as_mut_ptr(),
                info.as_mut_ptr(),
                constraint as *mut SuiteSparseInt,
            )
        };
        assert_eq!(res, super::CAMD_OK as SuiteSparseInt);
    }
}
