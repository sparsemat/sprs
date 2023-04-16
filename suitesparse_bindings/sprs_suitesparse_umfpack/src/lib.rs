use libc::c_void;
use sprs::errors::LinalgError;
use sprs::{CsMatI, SpIndex};
use suitesparse_umfpack_sys::*;

macro_rules! umfpack_impl {
    ($int: ty,
     $Context: ident,
     $Symbolic: ident,
     $Numeric: ident,
     $symbolic: ident,
     $numeric: ident,
     $solve: ident,
     $get_numeric: ident,
     $get_symbolic: ident,
     $get_lunz: ident,
     $free_numeric: ident,
     $free_symbolic: ident,
     ) => {

        struct $Symbolic(*const c_void);

        struct $Numeric(*const c_void);

        pub struct $Context {
            symbolic: $Symbolic,
            numeric: $Numeric,
            nrow: usize,
            ncol: usize,
            nnz: usize
        }

        impl $Context {
            pub unsafe fn new<N>(mat: &CsMatI<N, $int>) -> Self 
            where N: Default + Clone + Into<f64>,
            {
                // UMFPACK methods require CSC format and f64 data
                let mat: CsMatI<f64, $int> = mat.to_other_types().into_csc();

                // Get shape info
                let nrow = mat.rows();
                let ncol = mat.cols();
                let nnz = mat.nnz();

                // Get C-compatible raw pointers to column pointers, indices, and values
                let ap = mat.indptr().to_proper().as_ptr();
                let ai = mat.indices().as_ptr();
                let ax = mat.data().as_ptr();

                // Do symbolic factorization
                let symbolic_inner = (0 as *mut c_void) as *mut *mut c_void;
                unsafe {
                    $symbolic(
                        nrow as $int,
                        ncol as $int,
                        ap,
                        ai,
                        ax,
                        symbolic_inner,
                        0 as *const f64,  // Default settings
                        0 as *mut c_void  // Ignore info
                    );
                };
                let symbolic = $Symbolic(*symbolic_inner);

                // Do numeric factorization
                let numeric_inner = (0 as *mut c_void) as *mut *mut c_void;
                unsafe {
                    $numeric(
                        ap,
                        ai,
                        ax,
                        symbolic.0,
                        numeric_inner,
                        0 as *const f64,  // Default settings
                        0 as *mut c_void  // Ignore info
                    );
                };
                let numeric = $Numeric(*numeric_inner);

                Self {
                    symbolic: symbolic,
                    numeric: numeric,
                    nrow: nrow,
                    ncol: ncol,
                    nnz: nnz
                }
            }
        }
     };
}

umfpack_impl!(
    SuiteSparseInt,
    UmfpackDIContext,
    UmfpackDISymbolic,
    UmfpackDINumeric,
    umfpack_di_symbolic,
    umfpack_di_numeric,
    umfpack_di_solve,
    umfpack_di_get_numeric,
    umfpack_di_get_symbolic,
    umfpack_di_get_lunz,
    umfpack_di_free_numeric,
    umfpack_di_free_symbolic,
);

umfpack_impl!(
    SuiteSparseLong,
    UmfpackDLContext,
    UmfpackDLSymbolic,
    UmfpackDLNumeric,
    umfpack_dl_symbolic,
    umfpack_dl_numeric,
    umfpack_dl_solve,
    umfpack_dl_get_numeric,
    umfpack_dl_get_symbolic,
    umfpack_dl_get_lunz,
    umfpack_dl_free_numeric,
    umfpack_dl_free_symbolic,
);
