//! Partial interface (3rd party, unaffiliated) to `SuiteSparse`'s UMFPACK solver package,
//! covering essentials for solving problems of the form Ax=b for real A, and for
//! recovering LU decomposition components of A for other uses.
//!
//! This wrapper currently covers the double-int (DI) and double-long (DL) variations of
//! the underlying library, while several more variations (such as for complex data type)
//! exist in the underlying library but do not have wrappers here.

use core::ffi::c_void;
use core::ptr::{null, null_mut};
use sprs::{CsMatI, CsMatViewI, PermOwnedI};
use suitesparse_umfpack_sys::*;

macro_rules! umfpack_impl {
    ($int: ty,
     $Context: ident,
     $NumericComponents: ident,
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

        /// Wrapper of raw handle to guarantee proper drop procedure
        struct $Symbolic(*mut c_void);

        impl Drop for $Symbolic {
            fn drop(&mut self) {
                unsafe {$free_symbolic(core::ptr::addr_of_mut!(self.0).cast());}
            }
        }

        /// Wrapper of raw handle to guarantee proper drop procedure
        struct $Numeric(*mut c_void);

        impl Drop for $Numeric {
            fn drop(&mut self) {
                unsafe {$free_numeric(core::ptr::addr_of_mut!(self.0).cast());}
            }
        }

        /// Components of numeric factorization
        pub struct $NumericComponents {
            /// `L` matrix in CSC format
            pub l: CsMatI<f64, $int>,
            /// `U` matrix in CSR format
            pub u: CsMatI<f64, $int>,
            /// Row permutation
            pub p: PermOwnedI<$int>,
            /// Column permutation
            pub q: PermOwnedI<$int>,
            /// Inverse row scaling (divide rows of LU by these to recover PAQ)
            pub rs: Vec<f64>,
            /// Unknown usage; this quantity is not mentioned in underlying documentation but has distinct values, so we provide it here
            pub dx: Vec<f64>
        }

        /// Partial interface to `SuiteSparse`'s UMFPACK solver package.
        /// Provides LU factorization of A and solution of Ax=b using stored factorization.
        pub struct $Context {
            /// `A` matrix of system Ax=b
            a: CsMatI<f64, $int>,
            /// Opaque raw handle to symbolic factorization
            #[allow(dead_code)]  // We don't use this at the moment, but could extend the bindings to extract state
            symbolic: $Symbolic,
            /// Opaque raw handle to numeric factorization
            numeric: $Numeric,
            /// Number of rows in `A`
            nrow: usize,
            /// Number of cols in `A`
            ncol: usize,
            /// Number of nonzeroes in `A`
            nnz: usize,
        }

        impl $Context {

            /// Build a new stored factorization of matrix `A` into LU components
            /// with a stored C handle to an efficient (but opaque) direct solver.
            ///
            /// This factorization can be done for either square or rectangular `A` matrix,
            /// but can only be solved directly if `A` is square.
            pub fn new<N>(a: CsMatI<N, $int>) -> Self
            where N: Default + Clone + Into<f64>,
            {
                // UMFPACK methods require CSC format and f64 data
                let a: CsMatI<f64, $int> = a.to_other_types().into_csc();

                // Get shape info
                let nrow = a.rows();
                let ncol = a.cols();
                let nnz = a.nnz();

                // Get C-compatible raw pointers to column pointers, indices, and values
                let ap = a.indptr().to_proper().as_ptr();
                let ai = a.indices().as_ptr();
                let ax = a.data().as_ptr();

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
                    a,
                    symbolic,
                    numeric,
                    nrow,
                    ncol,
                    nnz,
                }
            }

            /// Get the shape of A like (`nrow`, `ncol`)
            pub fn shape(&self) -> (usize, usize) {
                (self.nrow as usize, self.ncol as usize)
            }

            /// Get the number of nonzero entries in A
            pub fn nnz(&self) -> usize {
                self.nnz as usize
            }

            /// Get a reference to the stored matrix,
            /// which may have had its data type converted from
            /// what was supplied.
            pub fn a(&self) -> CsMatViewI<f64, $int> {
                self.a.view()
            }

            /// Solve the system `Ax=b` for `x` given `b`,
            /// using the stored decomposition of the `A` matrix.
            ///
            /// # Panics
            ///
            /// * if `b` does not have the outer dimension of `A`
            /// * if `A` is not square
            pub fn solve(&self, b: &[f64]) -> Vec<f64> {
                // Check shape
                let (nrow, ncol) = self.shape();
                assert!(b.len() == nrow, "Input right-hand-side does not have the expected number of entries");
                assert!(nrow == ncol, "Solve can only be performed for square systems");

                // Allocate dense output vector
                let mut x = vec![0.0; ncol];

                // Get C-compatible raw pointers to column pointers, indices, and values
                let a = self.a();
                let ap = a.indptr().to_proper().as_ptr();
                let ai = a.indices().as_ptr();
                let ax = a.data().as_ptr();

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

            /// Get shape info about LU components
            /// * `lnz`: number of nonzero entries in `L`
            /// * `unz`: number of nonzero entries in `U`
            /// * `nrow`: number of rows in `L` and `U`
            /// * `ncol`: number of columns in `L` and `U`
            /// * `nz_udiag`: number of nonzeroes on the diagonal of `L` and `U`
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

            /// Get raw components of the numerical factorization of `A`
            /// * `l`: `L` matrix in CSC format
            /// * `u`: `U` matrix in CSR format
            /// * `p`: row permutation
            /// * `q`: column permutation
            /// * `rs`: inverse row scaling (divide rows of LU by these to recover PAQ)
            /// * `dx`: unknown; this quantity is not mentioned in underlying documentation but has distinct values, so we provide it here
            ///
            /// # Panics
            ///
            /// * if the extracted values and indices do not have the same length
            pub fn get_numeric(&self) -> $NumericComponents {
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
                let l = CsMatI::new_from_unsorted(shape, lp, lj, lx).unwrap();
                let u = CsMatI::new_from_unsorted_csc(shape, up, ui, ux).unwrap();
                let p = PermOwnedI::new(p);
                let q = PermOwnedI::new(q);

                $NumericComponents {
                    l,
                    u,
                    p,
                    q,
                    rs,
                    dx,
                }
            }
        }
    };
}

umfpack_impl!(
    SuiteSparseInt,
    UmfpackDIContext,
    UmfpackDINumericComponents,
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
    UmfpackDLNumericComponents,
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

    use crate::{UmfpackDIContext, UmfpackDLContext};

    #[test]
    fn umfpack_di() {
        let a = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1., 2., 21., 6., 6., 2., 2., 8.],
        );

        let ctx = UmfpackDIContext::new(a);

        let b = vec![1.0_f64; 4];

        let x = ctx.solve(&b[..]);
        println!("{:?}", x);

        let xsprs = CsVecI::new(4, vec![0, 1, 2, 3], x);

        let b_recovered = &ctx.a() * &xsprs;
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
        let _ = ctx.get_numeric();

        // FIXME: Once there's more functionality for doing row and column permutations, this needs a quantitative check
        // to make sure LUR = PAQ holds for the returned components
    }

    #[test]
    fn umfpack_dl() {
        let a = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1., 2., 21., 6., 6., 2., 2., 8.],
        );

        let ctx = UmfpackDLContext::new(a);

        let b = vec![1.0_f64; 4];

        let x = ctx.solve(&b[..]);
        println!("{:?}", x);

        let xsprs = CsVecI::new(4, vec![0, 1, 2, 3], x);

        let b_recovered = &ctx.a() * &xsprs;
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
        let _ = ctx.get_numeric();

        // FIXME: Once there's more functionality for doing row and column permutations, this needs a quantitative check
        // to make sure LUR = PAQ holds for the returned components
    }
}
