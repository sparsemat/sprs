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
    for ((_, sprow), mut drow) in iterator {
        for (ind, val) in sprow.iter() {
            drow[[ind]] = val.clone();
        }
    }

    Ok(())
}

