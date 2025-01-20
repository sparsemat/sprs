use ndarray::Array2;
use sprs::CsMatI;

type Mat = CsMatI<f32, u16, usize>;

fn dense_nnz(arr: &Array2<f32>) -> usize {
    arr.iter().map(|&f| usize::from(f != 0.0)).sum()
}

#[test]
#[should_panic(expected = "Index type is not large enough to hold")]
fn main() {
    // create matrix so that row index doesn't fit into column index type after transposition
    let mut x_sprs = Mat::zero((1_usize << 18, usize::from(1_u16 << 4)));
    x_sprs.insert(1_usize << 17, usize::from(1_u16 << 2), 1.0);

    // compute X^T * X in the sparse representation
    let x_t_sprs = x_sprs.transpose_view();
    let x_t_x_sprs = &x_t_sprs * &x_sprs;

    // compute X^T * X in the dense representation
    let x_dense = x_sprs.to_dense();
    let x_t_x_dense = x_dense.t().dot(&x_dense);

    // convert sparse result to dense representation for comparison
    let x_t_x_sprs_to_dense = x_t_x_sprs.to_dense();

    assert_eq!(x_sprs.nnz(), 1);
    assert_eq!(x_t_x_sprs.nnz(), 1);
    assert_eq!(dense_nnz(&x_dense), 1);
    assert_eq!(dense_nnz(&x_t_x_dense), 1);
    assert_eq!(dense_nnz(&x_t_x_sprs_to_dense), 1);
}
