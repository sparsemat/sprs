#![allow(non_camel_case_types)]
pub type ldl_int = std::ffi::c_int;

#[cfg(target_os = "windows")]
pub type ldl_long = i64;
#[cfg(not(target_os = "windows"))]
pub type ldl_long = std::ffi::c_long;
pub type ldl_double = std::ffi::c_double;

extern "C" {
    pub fn ldl_symbolic(
        n: ldl_int,
        ap: *const ldl_int,
        ai: *const ldl_int,
        lp: *mut ldl_int,
        parent: *mut ldl_int,
        lnz: *mut ldl_int,
        flag: *mut ldl_int,
        p: *const ldl_int,
        pinv: *const ldl_int,
    );

    pub fn ldl_numeric(
        n: ldl_int,
        ap: *const ldl_int,
        ai: *const ldl_int,
        ax: *const ldl_double,
        lp: *mut ldl_int,
        parent: *mut ldl_int,
        lnz: *mut ldl_int,
        li: *mut ldl_int,
        lx: *mut ldl_double,
        d: *mut ldl_double,
        y: *mut ldl_double,
        pattern: *mut ldl_int,
        flag: *mut ldl_int,
        p: *const ldl_int,
        pinv: *const ldl_int,
    ) -> ldl_int;

    pub fn ldl_lsolve(
        n: ldl_int,
        x: *mut ldl_double,
        lp: *const ldl_int,
        li: *const ldl_int,
        lx: *const ldl_double,
    );

    pub fn ldl_dsolve(n: ldl_int, x: *mut ldl_double, d: *const ldl_double);

    pub fn ldl_ltsolve(
        n: ldl_int,
        x: *mut ldl_double,
        lp: *const ldl_int,
        li: *const ldl_int,
        lx: *const ldl_double,
    );

    pub fn ldl_perm(
        n: ldl_int,
        x: *mut ldl_double,
        b: *const ldl_double,
        p: *const ldl_int,
    );

    pub fn ldl_permt(
        n: ldl_int,
        x: *mut ldl_double,
        b: *const ldl_double,
        p: *const ldl_int,
    );

    pub fn ldl_valid_perm(
        n: ldl_int,
        p: *const ldl_int,
        flag: *const ldl_int,
    ) -> ldl_int;

    pub fn ldl_valid_matrix(
        n: ldl_int,
        ap: *const ldl_int,
        ai: *const ldl_int,
    ) -> ldl_int;

    ////////////////////////////////////////////////////////////////////////////
    //////////////////          long version           /////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    pub fn ldl_l_symbolic(
        n: ldl_long,
        ap: *const ldl_long,
        ai: *const ldl_long,
        lp: *mut ldl_long,
        parent: *mut ldl_long,
        lnz: *mut ldl_long,
        flag: *mut ldl_long,
        p: *const ldl_long,
        pinv: *const ldl_long,
    );

    pub fn ldl_l_numeric(
        n: ldl_long,
        ap: *const ldl_long,
        ai: *const ldl_long,
        ax: *const ldl_double,
        lp: *mut ldl_long,
        parent: *mut ldl_long,
        lnz: *mut ldl_long,
        li: *mut ldl_long,
        lx: *mut ldl_double,
        d: *mut ldl_double,
        y: *mut ldl_double,
        pattern: *mut ldl_long,
        flag: *mut ldl_long,
        p: *const ldl_long,
        pinv: *const ldl_long,
    ) -> ldl_long;

    pub fn ldl_l_lsolve(
        n: ldl_long,
        x: *mut ldl_double,
        lp: *const ldl_long,
        li: *const ldl_long,
        lx: *const ldl_double,
    );

    pub fn ldl_l_dsolve(n: ldl_long, x: *mut ldl_double, d: *const ldl_double);

    pub fn ldl_l_ltsolve(
        n: ldl_long,
        x: *mut ldl_double,
        lp: *const ldl_long,
        li: *const ldl_long,
        lx: *const ldl_double,
    );

    pub fn ldl_l_perm(
        n: ldl_long,
        x: *mut ldl_double,
        b: *const ldl_double,
        p: *const ldl_long,
    );

    pub fn ldl_l_permt(
        n: ldl_long,
        x: *mut ldl_double,
        b: *const ldl_double,
        p: *const ldl_long,
    );

    pub fn ldl_l_valid_perm(
        n: ldl_long,
        p: *const ldl_long,
        flag: *const ldl_long,
    ) -> ldl_long;

    pub fn ldl_l_valid_matrix(
        n: ldl_long,
        ap: *const ldl_long,
        ai: *const ldl_long,
    ) -> ldl_long;
}

#[test]
fn sanity() {
    let n = 1;
    let ap = &[0, 1];
    let ai = &[0];
    let valid;
    unsafe {
        valid = ldl_valid_matrix(n, ap.as_ptr(), ai.as_ptr());
    }
    assert_eq!(valid, 1);
}
