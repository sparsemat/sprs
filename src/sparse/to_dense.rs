use super::CsMatViewI;
use crate::indexing::SpIndex;
use crate::Ix2;
///! Utilities for sparse-to-dense conversion
use ndarray::{ArrayViewMut, Axis};

/// Assign a sparse matrix into a dense matrix
///
/// The dense matrix will not be zeroed prior to assignment,
/// so existing values not corresponding to non-zeroes will be preserved.
pub fn assign_to_dense<N, I, Iptr>(
    mut array: ArrayViewMut<N, Ix2>,
    spmat: CsMatViewI<N, I, Iptr>,
) where
    N: Clone,
    I: SpIndex,
    Iptr: SpIndex,
{
    if spmat.cols() != array.shape()[1] {
        panic!("Dimension mismatch");
    }
    if spmat.rows() != array.shape()[0] {
        panic!("Dimension mismatch");
    }
    let outer_axis = if spmat.is_csr() { Axis(0) } else { Axis(1) };

    let iterator = spmat.outer_iterator().zip(array.axis_iter_mut(outer_axis));
    for (sprow, mut drow) in iterator {
        for (ind, val) in sprow.iter() {
            drow[[ind]] = val.clone();
        }
    }
}

#[cfg(test)]
mod test {
    use crate::test_data::{mat1, mat3};
    use crate::CsMat;
    use ndarray::{arr2, Array};

    #[test]
    fn to_dense() {
        let speye: CsMat<f64> = CsMat::eye(3);
        let mut deye = Array::zeros((3, 3));

        super::assign_to_dense(deye.view_mut(), speye.view());

        let res = Array::eye(3);
        assert_eq!(deye, res);

        let speye: CsMat<f64> = CsMat::eye_csc(3);
        let mut deye = Array::zeros((3, 3));

        super::assign_to_dense(deye.view_mut(), speye.view());

        assert_eq!(deye, res);

        let res = mat1().to_dense();
        let expected = arr2(&[
            [0., 0., 3., 4., 0.],
            [0., 0., 0., 2., 5.],
            [0., 0., 5., 0., 0.],
            [0., 8., 0., 0., 0.],
            [0., 0., 0., 7., 0.],
        ]);
        assert_eq!(expected, res);

        let res2 = mat3().to_dense();
        let expected2 = arr2(&[
            [0., 0., 3., 4.],
            [0., 0., 2., 5.],
            [0., 0., 5., 0.],
            [0., 8., 0., 0.],
            [0., 0., 0., 7.],
        ]);
        assert_eq!(expected2, res2);
    }
}
