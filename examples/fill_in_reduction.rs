extern crate ndarray;
///! This file demonstrates how it is possible to reduce the fill-in of a
///! symmetric sparse matrix during Cholesky decomposition.
extern crate sprs;

use ndarray::arr2;

fn main() {
    // 0 - A - 2 - 3
    // | \ | \ | / |
    // 7 - 5 - 6 - 4
    // | / | / | \ |
    // 8 - 9 - 1 - E
    #[rustfmt::skip]
    let triangles = arr2(
        &[[0, 7, 5],
          [0, 5, 10],
          [10, 5, 6],
          [10, 6, 2],
          [2, 6, 3],
          [3, 6, 4],
          [7, 8, 5],
          [5, 8, 9],
          [5, 9, 6],
          [6, 9, 1],
          [6, 1, 11],
          [6, 11, 4]],
    );
    let lap_mat = sprs::special_mats::tri_mesh_graph_laplacian(12, triangles);
    println!(
        "Lap mat nnz pattern:\n\n{}",
        sprs::visu::nnz_pattern_formatter(lap_mat.view()),
    );

    let ordering = sprs::linalg::cuthill_mckee(lap_mat.view());
    let perm_lap =
        sprs::transform_mat_papt(lap_mat.view(), ordering.perm.view());
    println!(
        "After Cuthill-McKee, profile is reduced:\n\n{}",
        sprs::visu::nnz_pattern_formatter(perm_lap.view()),
    );
}
