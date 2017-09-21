
extern crate libc;

#[allow(non_camel_case_types)]
pub type ldl_int = libc::c_int;
#[allow(non_camel_case_types)]
pub type ldl_long = libc::c_long;
#[allow(non_camel_case_types)]
pub type ldl_double = libc::c_double;

pub const LDL_MAIN_VERSION: usize = 2;
pub const LDL_SUB_VERSION: usize = 1;
pub const LDL_SUBSUB_VERSION: usize = 0;

extern "C" {
    pub fn ldl_symbolic(n: ldl_int,
                        ap: *mut ldl_int,
                        ai: *mut ldl_int,
                        lp: *mut ldl_int,
                        parent: *mut ldl_int,
                        lnz: *mut ldl_int,
                        flag: *mut ldl_int,
                        p: *mut ldl_int,
                        pinv: *mut ldl_int);

    pub fn ldl_numeric(n: ldl_int,
                       ap: *mut ldl_int,
                       ai: *mut ldl_int,
                       ax: *mut ldl_double,
                       lp: *mut ldl_int,
                       parent: *mut ldl_int,
                       lnz: *mut ldl_int,
                       li: *mut ldl_int,
                       lx: *mut ldl_double,
                       d: *mut ldl_double,
                       y: *mut ldl_double,
                       pattern: *mut ldl_int,
                       flag: *mut ldl_int,
                       p: *mut ldl_int,
                       pinv: *mut ldl_int);

    pub fn ldl_lsolve(n: ldl_int,
                      x: *mut ldl_double,
                      lp: *mut ldl_int,
                      li: *mut ldl_int,
                      lx: *mut ldl_double);

    pub fn ldl_dsolve(n: ldl_int,
                      x: *mut ldl_double,
                      d: *mut ldl_double);

    pub fn ldl_ltsolve(n: ldl_int,
                       x: *mut ldl_double,
                       lp: *mut ldl_int,
                       li: *mut ldl_int,
                       lx: *mut ldl_double);

    pub fn ldl_perm(n: ldl_int,
                    x: *mut ldl_double,
                    b: *mut ldl_double,
                    p: *mut ldl_int);

    pub fn ldl_permt(n: ldl_int,
                     x: *mut ldl_double,
                     b: *mut ldl_double,
                     p: *mut ldl_int);

    pub fn ldl_valid_perm(n: ldl_int,
                          p: *mut ldl_int,
                          flag: *mut ldl_int) -> ldl_int;

    pub fn ldl_valid_matrix(n: ldl_int,
                            ap: *mut ldl_int,
                            ai: *mut ldl_int) -> ldl_int;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
