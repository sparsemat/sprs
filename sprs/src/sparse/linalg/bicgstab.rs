//! Stabilized bi-conjugate gradient solver for solving Ax = b with x unknown. Suitable for non-symmetric matrices.
//! A simple, sparse-sparse, serial, un-preconditioned implementation.
//!
//! # References
//! The original paper, which is thoroughly paywalled but widely referenced:
//!
//! ```text
//! H. A. van der Vorst,
//! “Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems,”
//! SIAM Journal on Scientific and Statistical Computing, Jul. 2006, doi: 10.1137/0913035.
//! ```
//!
//! A useful discussion of computational cost and convergence characteristics for the CG
//! family of algorithms can be found in the paper that introduces QMRCGSTAB, in Table 1:
//!
//! ```text
//! T. F. Chan, E. Gallopoulos, V. Simoncini, T. Szeto, and C. H. Tong,
//! “A Quasi-Minimal Residual Variant of the Bi-CGSTAB Algorithm for Nonsymmetric Systems,”
//! SIAM J. Sci. Comput., vol. 15, no. 2, pp. 338–347, Mar. 1994, doi: 10.1137/0915023.
//! ```
//!
//! A less-paywalled pseudocode variant for this solver (as well as CG aand CGS) can be found at:
//! ```text
//! https://utminers.utep.edu/xzeng/2017spring_math5330/MATH_5330_Computational_Methods_of_Linear_Algebra_files/ln07.pdf
//! ```
//!
//! # Example
//! ```rust
//! use sprs::{CsMatI, CsVecI};
//! use sprs::linalg::bicgstab::BiCGSTAB;
//!
//! let a = CsMatI::new_csc(
//!     (4, 4),
//!     vec![0, 2, 4, 6, 8],
//!     vec![0, 3, 1, 2, 1, 2, 0, 3],
//!     vec![1.0, 2., 21., 6., 6., 2., 2., 8.],
//! );
//!
//! // Solve Ax=b
//! let tol = 1e-60;
//! let max_iter = 50;
//! let b = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0; 4]);
//! let x0 = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]);
//!
//! let res = BiCGSTAB::<'_, f64, _, _>::solve(
//!     a.view(),
//!     x0.view(),
//!     b.view(),
//!     tol,
//!     max_iter,
//! )
//! .unwrap();
//! let b_recovered = &a * &res.x();
//!
//! println!("Iteration count {:?}", res.iteration_count());
//! println!("Soft restart count {:?}", res.soft_restart_count());
//! println!("Hard restart count {:?}", res.hard_restart_count());
//!
//! // Make sure the solved values match expectation
//! for (input, output) in
//!     b.to_dense().iter().zip(b_recovered.to_dense().iter())
//! {
//!     assert!(
//!         (1.0 - input / output).abs() < tol,
//!         "Solved output did not match input"
//!     );
//! }
//! ```
//!
//! # Commentary
//! This implementation differs slightly from the common pseudocode variations in the following ways:
//! * Both soft-restart and hard-restart logic are present
//!   * Soft restart on `r` becoming perpendicular to `rhat`
//!   * Hard restart to check true error before claiming convergence
//! * Soft-restart logic uses a correct metric of perpendicularity instead of a magnitude heuristic
//!   * The usual method, which compares a fixed value to `rho`, does not capture the fact that the
//!   magnitude of `rho` will naturally decrease as the solver approaches convergence
//!   * This change eliminates the effect where the a soft restart is performed on every iteration for the last few
//!   iterations of any solve with a reasonable error tolerance
//! * Hard-restart logic provides some real guarantee of correctness
//!   * The usual implementations keep a cheap, but inaccurate, running estimate of the error
//!     * That decreases the cost of iterations by about half by eliminating a matrix-vector multiplication,
//!     but allows the estimate of error to drift numerically, which causes the solver to return claiming
//!     convergence when the solved output does not, in fact, match the input system
//!   * This change guarantees that the solver will not return claiming convergence unless the solution
//!   actually matches the input system, and will refresh its estimate of the error and continue iterations
//!   if it has reached a falsely-converged state, continuing until it either reaches true convergence or
//!   reaches maximum iterations
use crate::indexing::SpIndex;
use crate::sparse::{CsMatViewI, CsVecI, CsVecViewI};
use num_traits::One;

/// Stabilized bi-conjugate gradient solver
#[derive(Debug)]
pub struct BiCGSTAB<'a, T, I: SpIndex, Iptr: SpIndex> {
    // Configuration
    iteration_count: usize,
    soft_restart_threshold: T,
    soft_restart_count: usize,
    hard_restart_count: usize,
    // Problem statement: err = a * x - b
    err: T,
    a: CsMatViewI<'a, T, I, Iptr>,
    b: CsVecViewI<'a, T, I>,
    x: CsVecI<T, I>,
    // Intermediate vectors
    r: CsVecI<T, I>,
    rhat: CsVecI<T, I>, // Arbitrary w/ dot(rhat, r) != 0
    p: CsVecI<T, I>,
    // Intermediate scalars
    rho: T,
}

macro_rules! bicgstab_impl {
    ($T: ty) => {
        impl<'a, I: SpIndex, Iptr: SpIndex> BiCGSTAB<'a, $T, I, Iptr> {
            /// Initialize the solver with a fresh error estimate
            pub fn new(
                a: CsMatViewI<'a, $T, I, Iptr>,
                x0: CsVecViewI<'a, $T, I>,
                b: CsVecViewI<'a, $T, I>,
            ) -> Self {
                let r = &b - &(&a.view() * &x0.view()).view();
                let rhat = r.to_owned();
                let p = r.to_owned();
                let err = (&r).l2_norm();
                let rho = err * err;
                let x = x0.to_owned();
                Self {
                    iteration_count: 0,
                    soft_restart_threshold: 0.1 * <$T>::one(), // A sensible default
                    soft_restart_count: 0,
                    hard_restart_count: 0,
                    err,
                    a,
                    b,
                    x,
                    r,
                    rhat,
                    p,
                    rho,
                }
            }

            /// Attempt to solve the system to the given tolerance on normed error,
            /// returning an error if convergence is not achieved within the given
            /// number of iterations.
            pub fn solve(
                a: CsMatViewI<'a, $T, I, Iptr>,
                x0: CsVecViewI<'a, $T, I>,
                b: CsVecViewI<'a, $T, I>,
                tol: $T,
                max_iter: usize,
            ) -> Result<
                Box<BiCGSTAB<'a, $T, I, Iptr>>,
                Box<BiCGSTAB<'a, $T, I, Iptr>>,
            > {
                let mut solver = Self::new(a, x0, b);
                for _ in 0..max_iter {
                    solver.step();
                    if solver.err() < tol {
                        // Check true error, which may not match the running error estimate
                        // and either continue iterations or return depending on result.
                        solver.hard_restart();
                        if solver.err() < tol {
                            return Ok(Box::new(solver));
                        }
                    }
                }

                // If we ran past our iteration limit, error, but still return results
                Err(Box::new(solver))
            }

            /// Reset the reference direction `rhat` to be equal to `r`
            /// to prevent a singularity in `1 / rho`.
            pub fn soft_restart(&mut self) {
                self.soft_restart_count += 1;
                self.rhat = self.r.to_owned();
                self.rho = self.err * self.err; // Shortcut to (&self.r).squared_l2_norm();
                self.p = self.r.to_owned();
            }

            /// Recalculate the error vector from scratch using `a` and `b`.
            pub fn hard_restart(&mut self) {
                self.hard_restart_count += 1;
                // Recalculate true error
                self.r = &self.b - &(&self.a.view() * &self.x.view()).view();
                self.err = (&self.r).l2_norm();
                // Recalculate reference directions
                self.soft_restart();
                self.soft_restart_count -= 1; // Don't increment soft restart count for hard restarts
            }

            pub fn step(&mut self) -> $T {
                self.iteration_count += 1;

                // Gradient descent step
                let v = &self.a.view() * &self.p.view();
                let alpha = self.rho / ((&self.rhat).dot(&v));
                let h = &self.x + &self.p.map(|x| x * alpha); // latest estimate of `x`

                // Conjugate direction step
                let s = &self.r - &v.map(|x| x * alpha); // s = A*h
                let t = &self.a.view() * &s.view();
                let omega = t.dot(&s) / &t.squared_l2_norm();
                self.x = &h.view() + &s.map(|x| omega * x);

                // Check error
                self.r = &s - &t.map(|x| x * omega);
                self.err = (&self.r).l2_norm();

                // Prep for next pass
                let rho_prev = self.rho;
                self.rho = (&self.rhat).dot(&self.r);

                // Soft-restart if `rhat` is becoming perpendicular to `r`.
                if self.rho.abs() / (self.err * self.err)
                    < self.soft_restart_threshold
                {
                    self.soft_restart();
                } else {
                    let beta = (self.rho / rho_prev) * (alpha / omega);
                    self.p = &self.r
                        + (&self.p - &v.map(|x| x * omega)).map(|x| x * beta);
                }

                self.err
            }

            /// Set the minimum value of `rho` to trigger a soft restart
            pub fn with_restart_threshold(mut self, thresh: $T) -> Self {
                self.soft_restart_threshold = thresh;
                self
            }

            /// Iteration number
            pub fn iteration_count(&self) -> usize {
                self.iteration_count
            }

            /// The minimum value of `rho` to trigger a soft restart
            pub fn soft_restart_threshold(&self) -> $T {
                self.soft_restart_threshold
            }

            /// Number of soft restarts that have been done so far
            pub fn soft_restart_count(&self) -> usize {
                self.soft_restart_count
            }

            /// Number of soft restarts that have been done so far
            pub fn hard_restart_count(&self) -> usize {
                self.hard_restart_count
            }

            /// Latest estimate of normed error
            pub fn err(&self) -> $T {
                self.err
            }

            /// `dot(rhat, r)`, a measure of how well-conditioned the
            /// update to the gradient descent step direction will be.
            pub fn rho(&self) -> $T {
                self.rho
            }

            /// The problem matrix
            pub fn a(&self) -> CsMatViewI<'_, $T, I, Iptr> {
                self.a.view()
            }

            /// The latest solution vector
            pub fn x(&self) -> CsVecViewI<'_, $T, I> {
                self.x.view()
            }

            /// The objective vector
            pub fn b(&self) -> CsVecViewI<'_, $T, I> {
                self.b.view()
            }

            /// Latest residual error vector
            pub fn r(&self) -> CsVecViewI<'_, $T, I> {
                self.r.view()
            }

            /// Latest reference direction.
            /// `rhat` is arbitrary w/ dot(rhat, r) != 0,
            /// and is reset parallel to `r` when needed to avoid
            /// `1 / rho` becoming singular.
            pub fn rhat(&self) -> CsVecViewI<'_, $T, I> {
                self.rhat.view()
            }

            /// Gradient descent step direction, unscaled
            pub fn p(&self) -> CsVecViewI<'_, $T, I> {
                self.p.view()
            }
        }
    };
}

bicgstab_impl!(f64);
bicgstab_impl!(f32);

#[cfg(test)]
mod test {
    use super::*;
    use crate::CsMatI;

    #[test]
    fn test_bicgstab_f32() {
        let a = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1.0, 2., 21., 6., 6., 2., 2., 8.],
        );

        // Solve Ax=b
        let tol = 1e-18;
        let max_iter = 50;
        let b = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0; 4]);
        let x0 = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]);

        let res = BiCGSTAB::<'_, f32, _, _>::solve(
            a.view(),
            x0.view(),
            b.view(),
            tol,
            max_iter,
        )
        .unwrap();
        let b_recovered = &a * &res.x();

        println!("Iteration count {:?}", res.iteration_count());
        println!("Soft restart count {:?}", res.soft_restart_count());
        println!("Hard restart count {:?}", res.hard_restart_count());

        // Make sure the solved values match expectation
        for (input, output) in
            b.to_dense().iter().zip(b_recovered.to_dense().iter())
        {
            assert!(
                (1.0 - input / output).abs() < tol,
                "Solved output did not match input"
            );
        }
    }

    #[test]
    fn test_bicgstab_f64() {
        let a = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1.0, 2., 21., 6., 6., 2., 2., 8.],
        );

        // Solve Ax=b
        let tol = 1e-60;
        let max_iter = 50;
        let b = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0; 4]);
        let x0 = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]);

        let res = BiCGSTAB::<'_, f64, _, _>::solve(
            a.view(),
            x0.view(),
            b.view(),
            tol,
            max_iter,
        )
        .unwrap();
        let b_recovered = &a * &res.x();

        println!("Iteration count {:?}", res.iteration_count());
        println!("Soft restart count {:?}", res.soft_restart_count());
        println!("Hard restart count {:?}", res.hard_restart_count());

        // Make sure the solved values match expectation
        for (input, output) in
            b.to_dense().iter().zip(b_recovered.to_dense().iter())
        {
            assert!(
                (1.0 - input / output).abs() < tol,
                "Solved output did not match input"
            );
        }
    }
}
