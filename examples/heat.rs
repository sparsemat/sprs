
extern crate sprs;
extern crate ndarray;

/// Compute the discrete laplacian operator on a grid, assuming the
/// step size is 1.
/// We assume this operator operates on the C-order flattened version of
/// the grid.
///
/// This example shows how a relatively straightforward sparse matrix
/// can be constructed with a minimal number of allocations by directly
/// building up its sparse structure.
fn grid_laplacian(rows: usize, cols: usize) -> sprs::CsMatOwned<f64> {
    let nb_vert = rows * cols;
    let indptr = Vec::with_capacity(nb_vert + 1);
    let nnz = 5*nb_vert + 5;
    let indices = Vec::with_capacity(nnz);
    let data = Vec::with_capacity(nnz);
    let mut cumsum = 0;

    let add_elt = |i, j, x| {
        indices.push(i * rows + j);
        data.push(x);
        cumsum += 1;
    };

    for i in 0..rows {
        for k in 0..cols {
            indptr.push(cumsum);
            if i > 1 {
                add_elt(i - 1, j, 1.);
            }
            if j > 1 {
                add_elt(i, j - 1, 1.);
            }
            add_elt(i, j, -4.);
            if j + 1 < rows {
                add_elt(i, j + 1, 1.);
            }
            if i + 1 < rows {
                add_elt(i + 1, j, 1.);
            }
        }
    }

    indptr.push(cumsum);

    CsMatOwned::new((nb_vert, nb_vert), indptr, indices, data);
}
