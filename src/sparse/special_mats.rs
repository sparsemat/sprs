use ndarray::ArrayView2;
///! Common sparse matrices
use smallvec::SmallVec;

use crate::indexing::SpIndex;
use crate::sparse::CsMatI;

fn grid_neighbors4(
    row: usize,
    col: usize,
    shape: (usize, usize),
) -> SmallVec<[(usize, usize, f64); 4]> {
    let (rows, cols) = shape;
    let top_row = row == 0;
    let bottom_row = row + 1 == rows;

    let left_col = col == 0;
    let right_col = col + 1 == cols;
    let mut nb_neighbors = 0.;
    if !top_row {
        nb_neighbors += 1.;
    }
    if !bottom_row {
        nb_neighbors += 1.;
    }
    if !left_col {
        nb_neighbors += 1.;
    }
    if !right_col {
        nb_neighbors += 1.;
    }

    let mut res = SmallVec::new();
    if !top_row {
        res.push((row - 1, col, 1.));
    }
    if !left_col {
        res.push((row, col - 1, 1.));
    }
    res.push((row, col, -nb_neighbors));
    if !right_col {
        res.push((row, col + 1, 1.));
    }
    if !bottom_row {
        res.push((row + 1, col, 1.));
    }

    res
}

/// Compute the graph laplacian of a regular grid defined by its number of rows
/// and colums. Indexing of nodes in the grid is defined as the C-order raveling
/// of the grid locations.
pub fn grid2d_graph_laplacian<I>(shape: (usize, usize)) -> CsMatI<f64, I>
where
    I: SpIndex,
{
    let (rows, cols) = shape;
    let nb_vert = rows * cols;
    let mut indptr = Vec::with_capacity(nb_vert + 1);
    let nnz = 5 * nb_vert + 5;
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    let mut cumsum = I::zero();

    for i in 0..rows {
        for j in 0..cols {
            indptr.push(cumsum);
            let neighbors = grid_neighbors4(i, j, (rows, cols));
            for n in neighbors {
                indices.push(I::from_usize(n.0 * cols + n.1));
                data.push(n.2);
                cumsum += I::one();
            }
        }
    }

    indptr.push(cumsum);

    CsMatI::new_trusted(crate::CSR, (nb_vert, nb_vert), indptr, indices, data)
}

/// Compute the graph laplacian of a triangle mesh
pub fn tri_mesh_graph_laplacian<I>(
    nb_vertices: usize,
    triangles: ArrayView2<I>,
) -> CsMatI<f64, I>
// TODO generic scalar type
where
    I: SpIndex,
{
    assert!(triangles.shape()[1] == 3);
    let mut neighbors = vec![SmallVec::<[usize; 16]>::new(); nb_vertices];

    let mut insert_edge = |v0, v1| {
        let vert_neighbs: &mut SmallVec<[usize; 16]> = &mut neighbors[v0];
        if let Err(pos) = vert_neighbs.binary_search(&v1) {
            vert_neighbs.insert(pos, v1);
        }
    };
    for triangle in triangles.axis_iter(ndarray::Axis(0)) {
        let v0 = triangle[[0]].index();
        let v1 = triangle[[1]].index();
        let v2 = triangle[[2]].index();
        insert_edge(v0, v1);
        insert_edge(v1, v0);
        insert_edge(v0, v2);
        insert_edge(v2, v0);
        insert_edge(v1, v2);
        insert_edge(v2, v1);
    }
    let mut indptr = Vec::with_capacity(nb_vertices + 1);
    // Euler's formula gives an average vertex degree of 6 for triangle meshes
    let mut indices = Vec::with_capacity(7 * nb_vertices);
    let mut data = Vec::with_capacity(7 * nb_vertices);
    let mut nnz = I::zero();
    indptr.push(nnz);
    for (vert_ind, vert_neighbs) in neighbors.iter().enumerate() {
        let degree = vert_neighbs.len();
        nnz += I::from_usize(degree + 1);
        indptr.push(nnz);
        let mut below_diag = true;
        for neighbor in vert_neighbs {
            if below_diag && neighbor.index() > vert_ind {
                data.push(degree as f64);
                indices.push(I::from_usize(vert_ind));
                below_diag = false;
            }
            data.push(-1.);
            indices.push(I::from_usize(*neighbor));
        }
        if below_diag {
            data.push(degree as f64);
            indices.push(I::from_usize(vert_ind));
        }
    }
    CsMatI::new((nb_vertices, nb_vertices), indptr, indices, data)
}

#[cfg(test)]
mod test {
    use crate::sparse::CsMat;
    use ndarray::arr2;

    #[test]
    fn grid2d_graph_laplacian() {
        // 0 - 1 - 2 - 3
        // |   |   |   |
        // 4 - 5 - 6 - 7
        // |   |   |   |
        // 8 - 9 - a - b
        let lap = super::grid2d_graph_laplacian((3, 4));
        let a = 10;
        let b = 11;
        #[rustfmt::skip]
        let expected = CsMat::new(
            (12, 12),
            vec![0, 3, 7, 11, 14, 18, 23, 28, 32, 35, 39, 43, 46],
            vec![0, 1, 4,
                 0, 1, 2, 5,
                 1, 2, 3, 6,
                 2, 3, 7,
                 0, 4, 5, 8,
                 1, 4, 5, 6, 9,
                 2, 5, 6, 7, a,
                 3, 6, 7, b,
                 4, 8, 9,
                 5, 8, 9, a,
                 6, 9, a, b,
                 7, a, b,
            ],
            vec![-2., 1., 1.,
                 1., -3., 1., 1.,
                 1., -3., 1., 1.,
                 1., -2., 1.,
                 1., -3., 1., 1.,
                 1., 1., -4., 1., 1.,
                 1., 1., -4., 1., 1.,
                 1., 1., -3., 1.,
                 1., -2., 1.,
                 1., 1., -3., 1.,
                 1., 1., -3., 1.,
                 1., 1., -2.,
            ],
        );
        assert_eq!(lap, expected);
    }

    #[test]
    fn tri_mesh_graph_laplacian() {
        // 0 - 1 - 2
        // | \ | \ |
        // 3 - 4 - 5
        #[rustfmt::skip]
        let triangles = arr2(
            &[[0, 3, 4],
              [0, 4, 1],
              [1, 4, 5],
              [1, 5, 2]],
        );
        // Expected laplacian matrix (with x = -1)
        // | 3 x   x x   |
        // | x 4 x   x x |
        // |   x 2     x |
        // | x     2 x   |
        // | x x   x 4 x |
        // |   x x   x 3 |
        let x = -1.;
        #[rustfmt::skip]
        let expected = CsMat::new(
            (6, 6),
            vec![0, 4, 9, 12, 15, 20, 24],
            vec![0, 1, 3, 4,
                 0, 1, 2, 4, 5,
                 1, 2, 5,
                 0, 3, 4,
                 0, 1, 3, 4, 5,
                 1, 2, 4, 5,],
            vec![3., x, x, x,
                 x, 4., x, x, x,
                 x, 2., x,
                 x, 2., x,
                 x, x, x, 4., x,
                 x, x, x, 3.,],
        );
        let lap_mat = super::tri_mesh_graph_laplacian(6, triangles.view());
        assert_eq!(lap_mat, expected);
    }
}
