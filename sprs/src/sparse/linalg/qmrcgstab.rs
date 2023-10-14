//! Stabilized quasi-minimum-residual bi-conjugate gradient solver for solving Ax = b with x unknown. Suitable for non-symmetric matrices.
//! A simple, sparse-sparse, serial, un-preconditioned implementation.
//! 
//! Due to the use of the term `1 / (1 + err^2)^0.5`, the solver update here will
//! produce an error any time the error is less than around sqrt(<T>::EPSILON), where epsilon
//! is around 1e-16 for f64 and around 1e-7 for f32. Empirically, this sets a practical 
//! lower bound on tolerances of about 1e-6 for f64 and about 0.1 for f32 after which 
//! the solver will produce NaN values regardless of the quality of the inputs.
//! 
//! To summarize, it is numerically unstable even for well-conditioned inputs,
//! and should be used only for rough tolerances.
//!
//! # References
//!
//! The paper that introduces QMRCGSTAB2:
//!
//! ```text
//! T. F. Chan, E. Gallopoulos, V. Simoncini, T. Szeto, and C. H. Tong,
//! “A Quasi-Minimal Residual Variant of the Bi-CGSTAB Algorithm for Nonsymmetric Systems,”
//! SIAM J. Sci. Comput., vol. 15, no. 2, pp. 338–347, Mar. 1994, doi: 10.1137/0915023.
//! ```
//!
//! A useful discussion of computational cost and convergence characteristics for the CG
//! family of algorithms can be found in Table 1.
//!
//! # Commentary
//! This implementation differs slightly from the paper's pseudocode:
//! * The solver will restart if `rhat` becomes perpendicular to `r`, to prevent
//!   the update from becoming singular, or if any of the other intermediate scalars
//!   becomes NaN
//! * The true error is recalculated when the estimated error appears to be resolved,
//!   and the solver will either return or continue iterations based on the result
use crate::indexing::SpIndex;
use crate::sparse::{CsMatViewI, CsVecI, CsVecViewI};
use num_traits::One;

/// Stabilized quasi-minimum-residual bi-conjugate gradient solver
#[derive(Debug)]
pub struct QMRCGSTAB2<'a, T, I: SpIndex, Iptr: SpIndex> {
    // Configuration
    iteration_count: usize,
    restart_threshold: T,
    restart_count: usize,
    warm: bool,
    // Problem statement: err = a * x - b
    err: T,
    a: CsMatViewI<'a, T, I, Iptr>,
    b: CsVecViewI<'a, T, I>,
    x: CsVecI<T, I>,
    // Intermediate vectors
    r: CsVecI<T, I>,
    rhat: CsVecI<T, I>, // Arbitrary w/ dot(rhat, r) != 0
    p: CsVecI<T, I>,
    v: CsVecI<T, I>,
    d: CsVecI<T, I>,
    s: CsVecI<T, I>,
    // Intermediate scalars
    rho: T,
    alpha: T,
    omega: T,
    theta: T,
    eta: T,
    tau: T,
    tau_hat: T,
    theta_hat: T,
    eta_hat: T,
}

macro_rules! qmrcgstab_impl {
    ($T: ty) => {
        impl<'a, I: SpIndex, Iptr: SpIndex> QMRCGSTAB2<'a, $T, I, Iptr> {
            /// Initialize the solver with a fresh error estimate
            pub fn new(
                a: CsMatViewI<'a, $T, I, Iptr>,
                x0: CsVecViewI<'a, $T, I>,
                b: CsVecViewI<'a, $T, I>,
            ) -> Self {
                let r = &b - &(&a.view() * &x0.view()).view();
                let rhat = r.to_owned();
                let x = x0.to_owned();
                let p = CsVecI::<$T, I>::empty(r.dim());
                let v = CsVecI::<$T, I>::empty(r.dim());
                let d = CsVecI::<$T, I>::empty(r.dim());
                let s = CsVecI::<$T, I>::empty(r.dim());
                let err = (&r).l2_norm();
                let rho = 1.0;
                let alpha = 1.0;
                let omega = 1.0;
                let theta = 1.0;
                let eta = 1.0;
                let tau = err;
                let tau_hat = 0.0;
                let theta_hat = 0.0;
                let eta_hat = 0.0;
                let warm = false;
                Self {
                    iteration_count: 0,
                    restart_threshold: 0.1 * <$T>::one(), // A sensible default
                    restart_count: 0,
                    warm,
                    err,
                    a,
                    b,
                    x,
                    r,
                    rhat,
                    p,
                    v,
                    d,
                    s,
                    rho,
                    alpha,
                    omega,
                    theta,
                    eta,
                    tau,
                    tau_hat,
                    theta_hat,
                    eta_hat,
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
                Box<QMRCGSTAB2<'a, $T, I, Iptr>>,
                Box<QMRCGSTAB2<'a, $T, I, Iptr>>,
            > {
                let mut solver = Self::new(a, x0, b);
                for _ in 0..max_iter {
                    solver.step();

                    // if  {
                    //     solver.restart();
                    // }

                    if solver.err().is_nan() || solver.err() < tol {
                        // Check true error, which may not match the running error estimate
                        // and either continue iterations or return depending on result.
                        solver.restart();
                        if solver.err() < tol {
                            return Ok(Box::new(solver));
                        }
                    }
                }

                // If we ran past our iteration limit, error, but still return results
                Err(Box::new(solver))
            }

            /// Recalculate the residual and
            /// reset the reference direction `rhat` to be equal to `r`
            /// to prevent a singularity in `1 / rho`.
            pub fn restart(&mut self) {
                self.warm = false;
                self.restart_count += 1;

                // Recalculate true error and reference direction
                self.r = &self.b - &(&self.a.view() * &self.x.view()).view();
                self.err = (&self.r).l2_norm();
                self.rhat = self.r.to_owned();

                // Reset vectors
                self.p = CsVecI::<$T, I>::empty(self.r.dim());
                self.v = CsVecI::<$T, I>::empty(self.r.dim());
                self.d = CsVecI::<$T, I>::empty(self.r.dim());
                self.s = CsVecI::<$T, I>::empty(self.r.dim());

                // Reset scalars
                self.rho = 1.0;
                self.alpha = 1.0;
                self.omega = 1.0;
                self.theta = 1.0;
                self.eta = 1.0;
                self.tau_hat = 0.0;
                self.theta_hat = 0.0;
                self.eta_hat = 0.0;
            }

            pub fn step(&mut self) -> $T {
                self.iteration_count += 1;

                // Second quasi-minimization step,
                // reordered here to avoid changing the residual
                // after the error is reported at the end of the first
                // quasi-minimization step, or alternatively, to avoid
                // using an extra matrix-vector product per iteration
                // to check the true error at every iteration.
                if self.warm {
                    self.theta = self.err / self.tau_hat;
                    let c = 1.0 / (1.0 + self.theta.powi(2)).sqrt();
                    self.tau = self.tau_hat * self.theta * c;
                    self.eta = c.powi(2) * self.omega;

                    if self.theta.is_nan()
                        || c.is_nan()
                        || self.tau.is_nan()
                        || self.eta.is_nan()
                    {
                        self.restart();
                        return self.err;
                    }

                    self.d = &self.s
                        + &self.d.map(|x| {
                            x * (self.theta_hat.powi(2) * self.eta_hat
                                / self.omega)
                        });
                    self.x = &self.x + &self.d.map(|x| x * self.eta);
                }

                // Prep for first quasi-minimization step
                let mut rho_prev = self.rho;
                self.rho = (&self.rhat).dot(&self.r);

                // Restart if `rhat` is becoming perpendicular to `r`.
                if self.rho.abs() / (self.err * self.err)
                    < self.restart_threshold
                {
                    self.restart();
                    rho_prev = 1.0;
                }

                let beta = (self.rho * self.alpha) * (rho_prev * self.omega);
                self.p = &self.r
                    + (&self.p - &self.v.map(|x| x * self.omega))
                        .map(|x| x * beta);

                self.v = &self.a.view() * &self.p.view();
                self.alpha = self.rho / ((&self.rhat).dot(&self.v));
                self.s = &self.r - &self.v.map(|x| x * self.alpha);

                if self.alpha.is_nan() || beta.is_nan() || self.rho.is_nan() {
                    self.restart();
                    return self.err;
                }

                // First quasi-minimization step
                let snormsq = self.s.squared_l2_norm();
                self.theta_hat = snormsq / self.err;
                let c = 1.0 / (1.0 + self.theta_hat.powi(2)).sqrt();
                self.tau_hat = self.tau * self.theta_hat * c;
                self.eta_hat = c.powi(2) * self.alpha;

                if self.theta_hat.is_nan()
                    || self.tau_hat.is_nan()
                    || self.eta_hat.is_nan()
                    || c.is_nan()
                {
                    self.restart();
                    return self.err;
                }

                self.d = &self.p
                    + &self.d.map(|x| {
                        x * (self.theta.powi(2) * self.eta / self.alpha)
                    });
                self.x = &self.x + &self.d.map(|x| x * self.eta_hat); // latest estimate of `x`

                // Update residual
                let t = &self.a.view() * &self.s.view();
                self.omega = snormsq / t.dot(&self.s);

                if self.omega.is_nan() {
                    self.restart();
                    return self.err;
                }

                self.r = &self.s - &t.map(|x| x * self.omega);
                self.err = (&self.r).l2_norm();

                self.warm = true;
                self.err
            }

            /// Set the minimum value of `rho.abs() / err^2` to trigger a soft restart
            pub fn with_restart_threshold(mut self, thresh: $T) -> Self {
                self.restart_threshold = thresh;
                self
            }

            /// Iteration number
            pub fn iteration_count(&self) -> usize {
                self.iteration_count
            }

            /// The minimum value of `rho.abs() / err^2` to trigger a soft restart
            pub fn restart_threshold(&self) -> $T {
                self.restart_threshold
            }

            /// Number of soft restarts that have been done so far
            pub fn restart_count(&self) -> usize {
                self.restart_count
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

            /// Intermediate scalar `alpha`
            pub fn alpha(&self) -> $T {
                self.alpha
            }

            /// Intermediate scalar `omega`
            pub fn omega(&self) -> $T {
                self.omega
            }

            /// Intermediate scalar `theta`
            pub fn theta(&self) -> $T {
                self.theta
            }

            /// Intermediate scalar `eta`
            pub fn eta(&self) -> $T {
                self.eta
            }

            /// Intermediate scalar `tau`
            pub fn tau(&self) -> $T {
                self.tau
            }

            /// Intermediate scalar `tau_hat`
            pub fn tau_hat(&self) -> $T {
                self.tau_hat
            }

            /// Intermediate scalar `theta_hat`
            pub fn theta_hat(&self) -> $T {
                self.theta_hat
            }

            /// Intermediate scalar `eta_hat`
            pub fn eta_hat(&self) -> $T {
                self.eta_hat
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

            /// Intermediate vector `v`
            pub fn v(&self) -> CsVecViewI<'_, $T, I> {
                self.v.view()
            }

            /// Intermediate vector `d`
            pub fn d(&self) -> CsVecViewI<'_, $T, I> {
                self.d.view()
            }

            /// Intermediate vector `s`
            pub fn s(&self) -> CsVecViewI<'_, $T, I> {
                self.s.view()
            }
        }
    };
}

qmrcgstab_impl!(f64);
qmrcgstab_impl!(f32);

#[cfg(test)]
mod test {
    use super::*;
    use crate::CsMatI;

    #[test]
    fn test_qmrcgstab_f64() {
        let a = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1.0, 2., 21., 6., 6., 2., 2., 8.],
        );

        // Solve Ax=b
        let tol = 1e-6;
        let max_iter = 100;
        let b = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0; 4]);
        let x0 = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]);

        let res = QMRCGSTAB2::<'_, f64, _, _>::solve(
            a.view(),
            x0.view(),
            b.view(),
            tol,
            max_iter,
        )
        .unwrap();
        let b_recovered = &a * &res.x();

        println!("Iteration count {:?}", res.iteration_count());
        println!("Restart count {:?}", res.restart_count());

        // Make sure the solved values match expectation
        for (input, output) in
            b.to_dense().iter().zip(b_recovered.to_dense().iter())
        {
            assert!(
                (input - output).abs() <= tol,
                "Solved output did not match input"
            );
        }
    }

    #[test]
    fn test_qmrcgstab_f32() {
        let a = CsMatI::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1.0, 2., 21., 6., 6., 2., 2., 8.],
        );

        // Solve Ax=b
        let tol = 1e-1;
        let max_iter = 100;
        let b = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0; 4]);
        let x0 = CsVecI::new(4, vec![0, 1, 2, 3], vec![1.0, 1.0, 1.0, 1.0]);

        let res = QMRCGSTAB2::<'_, f32, _, _>::solve(
            a.view(),
            x0.view(),
            b.view(),
            tol,
            max_iter,
        )
        .unwrap();
        let b_recovered = &a * &res.x();

        println!("Iteration count {:?}", res.iteration_count());
        println!("Restart count {:?}", res.restart_count());

        // Make sure the solved values match expectation
        for (input, output) in
            b.to_dense().iter().zip(b_recovered.to_dense().iter())
        {
            assert!(
                (input - output).abs() <= tol,
                "Solved output did not match input"
            );
        }
    }
}
