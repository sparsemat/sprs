//! Random sparse matrix generation

use sprs::CsMatI;
use sprs::indexing::SpIndex;
use rand::prelude::*;
use rand::distributions::{Bernoulli, Distribution};

/// Generate a random sparse matrix structure matching the given density
pub fn rand_csr_structure<I>(shape: (usize, usize), density: f64)
    -> CsMatI<(), I>
where
    I: SpIndex,
{
    assert!(density >= 0. && density <= 1.);
    let mut rng = thread_rng();
    // Split the density into a row density and a col density, such that
    // `row_density * col_density == density`,
    // and `row_density / col_density ~= shape.0 / shape.1`
    // Hence `col_density * col_density = shape.1 * density / shape.0`
    if shape.0 == 0 || shape.1 == 0 {
        return CsMatI::new(
            shape, vec![I::from_usize(0)], Vec::new(), Vec::new()
        );
    }
    let col_density = (density * (shape.1 as f64) / (shape.0 as f64)).sqrt();
    let row_density = density / col_density;
    let exp_nnz =
        (density * (shape.0 as f64) * (shape.1 as f64)).ceil() as usize;
    let mut indptr = Vec::with_capacity(shape.0 + 1);
    let mut indices = Vec::with_capacity(exp_nnz);
    let mut data = Vec::with_capacity(exp_nnz);
    let mut nnz = 0;
    let row_dist = Bernoulli::new(row_density).unwrap();
    let col_dist = Bernoulli::new(col_density).unwrap();
    indptr.push(I::from_usize(nnz));
    for _row in 0..shape.0 {
        if row_dist.sample(&mut rng) {
            for col in 0..shape.1 {
                if col_dist.sample(&mut rng) {
                    nnz += 1;
                    indices.push(I::from_usize(col));
                    data.push(());
                }
            }
        }
        indptr.push(I::from_usize(nnz));
    }

    CsMatI::new(shape, indptr, indices, data)
}

#[cfg(test)]
mod tests {
    use sprs::CsMat;
    #[test]
    fn empty_random_mat() {
        let empty: CsMat<_> = super::rand_csr_structure((0, 0), 0.3);
        assert_eq!(empty.nnz(), 0);
    }

    #[test]
    fn random_csr_structure() {
        let mat: CsMat<_> = super::rand_csr_structure((100, 70), 0.3);
        // FIXME there is probably a probability for this test to fail,
        // the range should be tuned so that this probability is small enough
        assert!(mat.nnz() > (0.25f64 * 100. * 70.).ceil() as usize);
        assert!(mat.nnz() < (0.35f64 * 100. * 70.).floor() as usize);
    }
}
