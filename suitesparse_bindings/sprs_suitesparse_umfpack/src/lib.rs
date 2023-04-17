use core::ptr::{null, null_mut};
use libc::c_void;
use sprs::{CsMatI, PermOwnedI};
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
            ///
            #[allow(dead_code)]
            symbolic: $Symbolic,
            /// This isn't used directly at the moment, but we have to keep this around to free memory properly
            numeric: $Numeric,
            nrow: usize,
            ncol: usize,
            _nnz: usize,
        }

        impl $Context {
            pub fn new<N>(mat: CsMatI<N, $int>) -> Self
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
                let symbolic_inner = &mut (null_mut() as *mut c_void) as *mut *mut c_void;
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
                let symbolic = unsafe{$Symbolic(*symbolic_inner)};

                // Do numeric factorization
                let numeric_inner = &mut (null_mut() as *mut c_void) as *mut *mut c_void;
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
                let numeric = unsafe{$Numeric(*numeric_inner)};

                Self {
                    mat: mat,
                    symbolic: symbolic,
                    numeric: numeric,
                    nrow: nrow,
                    ncol: ncol,
                    _nnz: nnz
                }
            }

            pub fn shape(&self) -> (usize, usize) {
                (self.nrow as usize, self.ncol as usize)
            }

            pub fn nnz(&self) -> usize {
                self._nnz as usize
            }

            pub fn a(&self) -> &CsMatI<f64, $int> {
                &self.mat
            }

            pub fn solve(&self, b: &[f64]) -> Vec<f64> {
                // Check shape
                let (nrow, ncol) = self.shape();
                assert!(b.len() == nrow, "Input right-hand-side does not have the expected number of entries");
                assert!(nrow == ncol, "Solve can only be performed for square systems");

                // Allocate dense output vector
                let mut x = vec![0.0; ncol];

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

            pub fn get_lunz(&self) -> ($int, $int, $int, $int, $int) {
                let mut lnz: $int = 0;  // Total number of nonzero entries in L
                let mut unz: $int = 0;  // Total number of nonzero entries in U
                let mut nrow: $int = 0;  // Number of rows in both L and U (should match input matrix)
                let mut ncol: $int = 0;  // Number of cols in both L and U (should match input matrix)
                let mut nz_udiag: $int = 0;  // Number of nonzero entries on the diagonal of each L and U

                unsafe {
                    $get_lunz(
                        &mut lnz as *mut $int,
                        &mut unz as *mut $int,
                        &mut nrow as *mut $int,
                        &mut ncol as *mut $int,
                        &mut nz_udiag as *mut $int,
                        self.numeric.0
                    );
                }

                (lnz, unz, nrow, ncol, nz_udiag)
            }


            pub fn get_numeric(&self) -> (CsMatI<f64, $int>, CsMatI<f64, $int>, PermOwnedI<$int>, PermOwnedI<$int>, Vec<f64>, Vec<f64>) {
                // Get shape info that tells us how much to allocate
                let (lnz, unz, nrow, ncol, _) = self.get_lunz();
                let n_inner = nrow.min(ncol) as usize;
                let shape = (nrow as usize, ncol as usize);

                // Allocate for the return values
                let mut lp = vec![0 as $int; (nrow + 1) as usize];
                let mut lj = vec![0 as $int; lnz as usize];
                let mut lx = vec![0.0_f64; lnz as usize];

                let mut up = vec![0 as $int; (ncol + 1) as usize];
                let mut ui = vec![0 as $int; unz as usize];
                let mut ux = vec![0.0_f64; unz as usize];

                let mut rs = vec![0.0_f64; nrow as usize];
                let mut dx = vec![0.0_f64; n_inner as usize];

                let mut p = vec![0 as $int; nrow as usize];
                let mut q = vec![0 as $int; ncol as usize];

                // Extract the values from the opaque inner representation
                let do_recip = 0 as $int;  // Divide the scaling factors (the default behavior)
                unsafe {
                    $get_numeric(
                        lp[..].as_mut_ptr(),
                        lj[..].as_mut_ptr(),
                        lx[..].as_mut_ptr(),
                        up[..].as_mut_ptr(),
                        ui[..].as_mut_ptr(),
                        ux[..].as_mut_ptr(),
                        p[..].as_mut_ptr(),
                        q[..].as_mut_ptr(),
                        dx[..].as_mut_ptr(),
                        &do_recip as *const $int,
                        rs[..].as_mut_ptr(),
                        self.numeric.0
                    );
                }

                // Pack results into sparse matrix structures
                let l = CsMatI::new(shape, lp, lj, lx);
                let u = CsMatI::new_csc(shape, up, ui, ux);
                let p = PermOwnedI::new(p);
                let q = PermOwnedI::new(q);

                return (l, u, p, q, dx, rs)
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
    umfpack_dl_get_lunz,
    umfpack_dl_free_numeric,
    umfpack_dl_free_symbolic,
);

#[cfg(test)]
mod tests {
    use sprs::{CsMatI, CsVecI};

    use crate::UmfpackDIContext;

    #[test]
    fn umfpack_di() {
        let mat = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1., 2., 21., 6., 6., 2., 2., 8.],
        );

        let ctx = UmfpackDIContext::new(mat);

        let b = vec![1.0_f64; 4];

        let _x = ctx.solve(&b[..]);
        println!("{:?}", _x);

        let x = CsVecI::new(4, vec![0_i32, 1_i32, 2_i32, 3_i32], _x);

        let b_recovered = ctx.a() * &x;
        println!("{:?}", b_recovered);

        // Make sure the solved values match expectation
        for (input, output) in
            b.into_iter().zip(b_recovered.to_dense().into_iter())
        {
            assert!(
                (1.0 - input / output).abs() < 1e-14,
                "Solved output did not match input"
            );
        }

        // Smoketest get_numeric - can we get the LU components out without a segfault?
        let (l, u, p, q, dx, rs) = ctx.get_numeric();
    }
}
