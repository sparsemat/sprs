///! Utilities for sparse-to-dense conversion

use ndarray::{ArrayViewMut, Axis};
use errors::SprsError;
use ::{CsMatView, SpRes};
use ::Ix2;

/// Assign a sparse matrix into a dense matrix
///
/// The dense matrix will not be zeroed prior to assignment,
/// so existing values not corresponding to non-zeroes will be preserved.
pub fn assign_to_dense<N>(mut array: ArrayViewMut<N, Ix2>,
                          spmat: CsMatView<N>
                         ) -> SpRes<()>
where N: Clone
{
    if spmat.cols() != array.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    if spmat.rows() != array.shape()[0] {
        return Err(SprsError::IncompatibleDimensions);
    }
    let outer_axis = if spmat.is_csr() { Axis(0) } else { Axis(1) };

    let iterator = spmat.outer_iterator().zip(array.axis_iter_mut(outer_axis));
    for (sprow, mut drow) in iterator {
        for (ind, val) in sprow.iter() {
            drow[[ind]] = val.clone();
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use ndarray::{OwnedArray, arr2};
    use ::CsMatOwned;
    use test_data::{mat1};

    #[test]
    fn to_dense() {
        let speye: CsMatOwned<f64> = CsMatOwned::eye(3);
        let mut deye = OwnedArray::zeros((3, 3));

        super::assign_to_dense(deye.view_mut(), speye.view()).unwrap();

        let res = OwnedArray::eye(3);
        assert_eq!(deye, res);

        let speye: CsMatOwned<f64> = CsMatOwned::eye_csc(3);
        let mut deye = OwnedArray::zeros((3, 3));

        super::assign_to_dense(deye.view_mut(), speye.view()).unwrap();

        assert_eq!(deye, res);

        let res = mat1().to_dense();
        let expected = arr2(&[[0., 0., 3., 4., 0.],
                              [0., 0., 0., 2., 5.],
                              [0., 0., 5., 0., 0.],
                              [0., 8., 0., 0., 0.],
                              [0., 0., 0., 7., 0.]]);
        assert_eq!(expected, res);
    }
}
