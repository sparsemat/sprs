use crate::indexing::SpIndex;
use crate::sparse::prelude::*;

/// Compute the Kronecker product between two matrices
///
/// The storage order of the product will be based on the first matrix.
/// This method will perform a clone if the two matrices
/// have different storage order
///
/// # Panics
///
/// * if indices are out of bounds for its type
#[must_use]
pub fn kronecker_product<
    N: num_traits::Num + Copy + Default,
    I: SpIndex,
    Iptr: SpIndex,
>(
    mut a: CsMatViewI<N, I, Iptr>,
    mut b: CsMatViewI<N, I, Iptr>,
) -> CsMatI<N, I, Iptr> {
    if a.storage() == b.storage() {
        let was_csc = a.is_csc();
        if was_csc {
            a.transpose_mut();
            b.transpose_mut();
        }
        let nnz = a.nnz() * b.nnz();
        let a_shape = a.shape();
        let b_shape = b.shape();
        let shape = (a_shape.0 * b_shape.0, a_shape.1 * b_shape.1);
        let mut values = Vec::with_capacity(nnz);
        let mut indices = Vec::with_capacity(nnz);
        let mut indptr = Vec::with_capacity(shape.1 + 1);

        let mut element_count = Iptr::zero();
        indptr.push(element_count);
        for a in a.outer_iterator() {
            for b in b.outer_iterator() {
                for (ai, &a) in a.iter() {
                    for (bi, &b) in b.iter() {
                        indices.push(I::from(ai * b_shape.1 + bi).unwrap());
                        element_count += Iptr::one();
                        values.push(a * b);
                    }
                }
                indptr.push(element_count);
            }
        }
        let mut mat = CsMatBase::new(shape, indptr, indices, values);
        debug_assert_eq!(mat.nnz(), nnz);
        if was_csc {
            mat.transpose_mut()
        }
        mat
    } else {
        kronecker_product(a, b.to_owned().to_other_storage().view())
    }
}

#[test]
fn test_kronecker_product() {
    let mut a = TriMat::new((2, 3));
    a.add_triplet(0, 1, 2);
    a.add_triplet(0, 2, 3);
    a.add_triplet(1, 0, 6);
    a.add_triplet(1, 2, 8);
    let a = a.to_csr();

    let mut b = TriMat::new((3, 2));
    b.add_triplet(0, 0, 1);
    b.add_triplet(1, 0, 2);
    b.add_triplet(2, 0, 3);
    b.add_triplet(2, 1, -3);
    let b = b.to_csr();

    let check = |c: CsMatView<i32>| {
        for (&n, (j, i)) in c.iter() {
            match (j, i) {
                (0, 2) => assert_eq!(n, 2),
                (0, 4) => assert_eq!(n, 3),
                (1, 2) => assert_eq!(n, 4),
                (1, 4) => assert_eq!(n, 6),
                (2, 2) => assert_eq!(n, 6),
                (2, 3) => assert_eq!(n, -6),
                (2, 4) => assert_eq!(n, 9),
                (2, 5) => assert_eq!(n, -9),
                (3, 0) => assert_eq!(n, 6),
                (3, 4) => assert_eq!(n, 8),
                (4, 0) => assert_eq!(n, 12),
                (4, 4) => assert_eq!(n, 16),
                (5, 0) => assert_eq!(n, 18),
                (5, 1) => assert_eq!(n, -18),
                (5, 4) => assert_eq!(n, 24),
                (5, 5) => assert_eq!(n, -24),
                _ => panic!("index ({},{}) should be 0, found {}", j, i, n),
            }
        }
    };

    let c = kronecker_product(a.view(), b.view());
    check(c.view());
    let b = b.to_csc();
    let c = kronecker_product(a.view(), b.view());
    check(c.view());
    let a = a.to_csc();
    let c = kronecker_product(a.view(), b.view());
    check(c.view());
    let b = b.to_csr();
    let c = kronecker_product(a.view(), b.view());
    check(c.view());
}
