extern crate ndarray;
///! This file demonstrates how it is possible to reduce the fill-in of a
///! symmetric sparse matrix during Cholesky decomposition.
extern crate sprs;
extern crate tobj;

use ndarray::{arr2, ArrayView2};
use std::env;
use std::path::Path;

fn small_lap_mat() -> sprs::CsMat<f64> {
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
    let lap_mat =
        sprs::special_mats::tri_mesh_graph_laplacian(12, triangles.view());
    lap_mat
}

fn lap_mat_from_obj(path: &str) -> Result<sprs::CsMat<f64>, tobj::LoadError> {
    let (objects, _materials) = tobj::load_obj(&Path::new(path))?;
    for obj in objects {
        let nb_triangles = obj.mesh.indices.len() / 3;
        let nb_vertices = obj.mesh.positions.len() / 3;
        let triangles =
            ArrayView2::from_shape((nb_triangles, 3), &obj.mesh.indices[..])
                .unwrap();
        let lap_mat = sprs::special_mats::tri_mesh_graph_laplacian(
            nb_vertices,
            triangles,
        );
        return Ok(lap_mat.to_other_types());
    }
    eprintln!("No model found in obj file");
    Err(tobj::LoadError::ReadError)
}

fn main() -> Result<(), tobj::LoadError> {
    let mut args = env::args();
    args.next();
    let lap_mat = if let Some(path) = args.next() {
        lap_mat_from_obj(&path)?
    } else {
        small_lap_mat()
    };

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
    Ok(())
}
