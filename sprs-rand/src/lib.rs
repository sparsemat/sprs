//! Random sparse matrix generation

use crate::rand::distributions::{Bernoulli, Distribution};
use crate::rand::Rng;
use sprs::indexing::SpIndex;
use sprs::{CsMat, CsMatI};

/// Re-export [`rand`](https://docs.rs/rand/0.7.3/rand/)
/// for version compatibility
pub mod rand {
    pub use rand::*;
}

/// Re-export [`rand_distr`](https://docs.rs/rand_distr/0.2.2/rand_distr)
/// for version compatibility
pub mod rand_distr {
    pub use rand_distr::*;
}

/// Generate a random sparse matrix matching the given density and sampling
/// the values of its non-zero elements from the provided distribution.
pub fn rand_csr<R, N, D, I>(
    rng: &mut R,
    dist: D,
    shape: (usize, usize),
    density: f64,
) -> CsMatI<N, I>
where
    R: Rng + ?Sized,
    D: Distribution<N>,
    N: Copy,
    I: SpIndex,
{
    assert!(density >= 0. && density <= 1.);
    let exp_nnz =
        (density * (shape.0 as f64) * (shape.1 as f64)).ceil() as usize;
    let mut indptr = Vec::with_capacity(shape.0 + 1);
    let mut indices = Vec::with_capacity(exp_nnz);
    let mut data = Vec::with_capacity(exp_nnz);
    let mut nnz = 0;
    let struct_dist = Bernoulli::new(density).unwrap();
    indptr.push(I::from_usize(nnz));
    for _row in 0..shape.0 {
        for col in 0..shape.1 {
            if struct_dist.sample(rng) {
                nnz += 1;
                indices.push(I::from_usize(col));
                data.push(dist.sample(rng));
            }
        }
        indptr.push(I::from_usize(nnz));
    }

    CsMatI::new(shape, indptr, indices, data)
}

/// Convenient wrapper for the common case of sampling a matrix with standard
/// normal distribution of the nnz values, using the thread rng for convenience.
pub fn rand_csr_std(shape: (usize, usize), density: f64) -> CsMat<f64> {
    let mut rng = rand::thread_rng();
    rand_csr(&mut rng, crate::rand_distr::StandardNormal, shape, density)
}

#[cfg(test)]
mod tests {
    use rand::distributions::Standard;
    use rand::SeedableRng;
    use sprs::CsMat;

    #[test]
    fn empty_random_mat() {
        let mut rng = rand::thread_rng();
        let empty: CsMat<f64> =
            super::rand_csr(&mut rng, Standard, (0, 0), 0.3);
        assert_eq!(empty.nnz(), 0);
    }

    #[test]
    fn random_csr() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let mat: CsMat<f64> =
            super::rand_csr(&mut rng, Standard, (100, 70), 0.3);
        assert!(mat.density() > 0.25);
        assert!(mat.density() < 0.35);

        let mat: CsMat<f64> =
            super::rand_csr(&mut rng, Standard, (1, 10000), 0.3);
        assert!(mat.density() > 0.28);
        assert!(mat.density() < 0.32);
    }

    #[test]
    fn random_csr_std() {
        let mat = super::rand_csr_std((100, 1000), 0.2);
        assert_eq!(mat.shape(), (100, 1000));
        // Not checking the density as I have no control over the seed
        // Checking the mean nnz value should be safe though
        assert!(
            mat.data().iter().sum::<f64>().abs() / (mat.data().len() as f64)
                < 0.05
        );
    }
}
