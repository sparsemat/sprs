use core::ptr::{null, null_mut};
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

        struct $Symbolic(*mut c_void);

        impl Drop for $Symbolic {
            fn drop(&mut self) {
                unsafe {$free_symbolic(&mut self.0 as *mut *mut c_void);}
            }
        }

        struct $Numeric(*mut c_void);

        impl Drop for $Numeric {
            fn drop(&mut self) {
                unsafe {$free_numeric(&mut self.0 as *mut *mut c_void);}
            }
        }

        pub struct $Context {
            mat: CsMatI<f64, $int>,
            symbolic: $Symbolic,
            numeric: $Numeric,
            nrow: usize,
            ncol: usize,
            nnz: usize,
        }

        impl $Context {
            pub unsafe fn new<N>(mat: CsMatI<N, $int>) -> Self
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
                        null() as *const f64,  // Default settings
                        null_mut() as *mut c_void  // Ignore info
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
                        null() as *const f64,  // Default settings
                        null_mut() as *mut c_void  // Ignore info
                    );
                };
                let numeric = $Numeric(*numeric_inner);

                Self {
                    mat: mat,
                    symbolic: symbolic,
                    numeric: numeric,
                    nrow: nrow,
                    ncol: ncol,
                    nnz: nnz
                }
            }

            pub unsafe fn solve(&self, b: &[f64]) -> Vec<f64> {
                // Check shape
                let n = self.nrow * self.ncol;
                assert!(b.len() == n, "Input right-hand-side does not have the expected number of entries");
                assert!(self.nrow == self.ncol, "Solve can only be performed for square systems");

                // Allocate dense output vector
                let mut x = vec![0.0; n];

                // Get C-compatible raw pointers to column pointers, indices, and values
                let ap = self.mat.indptr().to_proper().as_ptr();
                let ai = self.mat.indices().as_ptr();
                let ax = self.mat.data().as_ptr();

                // Do the linear solve using pre-computed LU factors and permutation
                unsafe {
                    $solve(
                        0,  // sys=0 for Ax=b problem type
                        ap,
                        ai,
                        ax,
                        x[..].as_mut_ptr(),
                        b.as_ptr(),
                        self.numeric.0,
                        null() as *const f64,  // Default settings
                        null_mut() as *mut c_void  // Ignore info
                    )
                };

                x
            }

            pub unsafe fn get_lunz(&self) -> ($int, $int, $int, $int, $int) {
                let mut lnz: $int = 0;
                let mut unz: $int = 0;
                let mut nrow: $int = 0;
                let mut ncol: $int = 0;
                let mut nz_udiag: $int = 0;

                unsafe {
                    $get_lunz(
                        &mut lnz as *mut $int,
                        &mut unz as *mut $int,
                        &mut nrow as *mut $int,
                        &mut ncol as *mut $int,
                        &mut nz_udiag as *mut $int,
                        self.numeric.0 as *const c_void);
                }

                (lnz, unz, nrow, ncol, nz_udiag)
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
