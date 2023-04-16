use super::{SuiteSparseInt, SuiteSparseLong};
use libc::{c_double, c_void};

// int umfpack_di_get_numeric
// (
//     int Lp [ ],
//     int Lj [ ],
//     double Lx [ ],
//     int Up [ ],
//     int Ui [ ],
//     double Ux [ ],
//     int P [ ],
//     int Q [ ],
//     double Dx [ ],
//     int *do_recip,
//     double Rs [ ],
//     void *Numeric
// ) ;

extern "C" {
    fn umfpack_di_get_numeric(
        Lp: *mut SuiteSparseInt,
        Lj: *mut SuiteSparseInt,
        Lx: *mut c_double,
        Up: *mut SuiteSparseInt,
        Ui: *mut SuiteSparseInt,
        Ux: *mut c_double,
        P: *mut SuiteSparseInt,
        Q: *mut SuiteSparseInt,
        Dx: *mut c_double,
        do_recip: *mut SuiteSparseInt,
        Rs: *mut c_double,
        Numeric: *const c_void,
    ) -> SuiteSparseInt;
}

pub fn umfpack_di_get_numeric_wrapper(
    lp: &mut [SuiteSparseInt],
    lj: &mut [SuiteSparseInt],
    lx: &mut [c_double],
    up: &mut [SuiteSparseInt],
    ui: &mut [SuiteSparseInt],
    ux: &mut [c_double],
    p: &mut [SuiteSparseInt],
    q: &mut [SuiteSparseInt],
    dx: &mut [c_double],
    do_recip: &mut SuiteSparseInt,
    rs: &mut [c_double],
    numeric: *const c_void,
) -> SuiteSparseInt {
    unsafe {
        umfpack_di_get_numeric(
            lp.as_mut_ptr(),
            lj.as_mut_ptr(),
            lx.as_mut_ptr(),
            up.as_mut_ptr(),
            ui.as_mut_ptr(),
            ux.as_mut_ptr(),
            p.as_mut_ptr(),
            q.as_mut_ptr(),
            dx.as_mut_ptr(),
            do_recip as *mut SuiteSparseInt,
            rs.as_mut_ptr(),
            numeric,
        )
    }
}
