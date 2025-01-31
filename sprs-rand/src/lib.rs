//! Random sparse matrix generation

use crate::rand::distr::Distribution;
use crate::rand::Rng;
use crate::rand::SeedableRng;
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
    assert!((0.0..=1.0).contains(&density));
    let exp_nnz =
        (density * (shape.0 as f64) * (shape.1 as f64)).ceil() as usize;
    let mut indptr = Vec::with_capacity(shape.0 + 1);
    let mut indices = Vec::with_capacity(exp_nnz);
    let mut data = Vec::with_capacity(exp_nnz);
    // sample row indices
    for _ in 0..exp_nnz {
        indices.push(I::from_usize(rng.random_range(0..shape.0)));
        // Note: there won't be any correspondence between the data
        // sampled here and the row sampled before, but this does not matter
        // as we are sampling.
        data.push(dist.sample(rng));
    }
    indices.sort_unstable();
    indptr.push(I::from_usize(0));
    let mut count = 0;
    for &row in &indices {
        while indptr.len() != row.index() + 1 {
            indptr.push(I::from_usize(count));
        }
        count += 1;
    }
    while indptr.len() != shape.0 + 1 {
        indptr.push(I::from_usize(count));
    }
    assert_eq!(indptr.last().unwrap().index(), exp_nnz);
    indices.clear();
    for row in 0..shape.0 {
        let start = indptr[row].index();
        let end = indptr[row + 1].index();
        for _ in start..end {
            loop {
                let col = I::from_usize(rng.random_range(0..shape.1));
                let loc = indices[start..].binary_search(&col);
                match loc {
                    Ok(_) => {
                        continue;
                    }
                    Err(loc) => {
                        indices.insert(start + loc, col);
                        break;
                    }
                }
            }
        }
        indices[start..end].sort_unstable();
    }

    CsMatI::new(shape, indptr, indices, data)
}

/// Convenient wrapper for the common case of sampling a matrix with standard
/// normal distribution of the nnz values, using a lightweight rng.
pub fn rand_csr_std(shape: (usize, usize), density: f64) -> CsMat<f64> {
    let mut rng = rand_pcg::Pcg64Mcg::from_rng(&mut rand::rng());
    rand_csr(&mut rng, crate::rand_distr::StandardNormal, shape, density)
}

#[cfg(test)]
mod tests {
    use rand::distr::StandardUniform;
    use rand::SeedableRng;
    use sprs::CsMat;

    #[test]
    fn empty_random_mat() {
        let mut rng = rand::rng();
        let empty: CsMat<f64> =
            super::rand_csr(&mut rng, StandardUniform, (0, 0), 0.3);
        assert_eq!(empty.nnz(), 0);
    }

    #[test]
    fn random_csr() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let mat: CsMat<f64> =
            super::rand_csr(&mut rng, StandardUniform, (100, 70), 0.3);
        assert!(mat.density() > 0.25);
        assert!(mat.density() < 0.35);

        let mat: CsMat<f64> =
            super::rand_csr(&mut rng, StandardUniform, (1, 10000), 0.3);
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
