/// Functions dealing with symmetric sparse matrices

use sparse::csmat::CsMat;

pub fn is_symmetric<N: Clone + Copy + PartialEq>(mat: &CsMat<N>) -> bool {
    if mat.rows() != mat.cols() {
        return false;
    }
    let n = mat.rows();
    for (outer_ind, vec) in mat.outer_iterator() {
        for (inner_ind, value) in vec.iter() {
            match mat.at_outer_inner(&(inner_ind, outer_ind)) {
                None => return false,
                Some(transposed_val) => if transposed_val != value {
                    return false;
                }
            }
        }
    }
    true
}
