
extern crate sprs;
extern crate ndarray;

type VecViewMut<'a, T> = ndarray::ArrayViewMut<'a, T, ndarray::Ix>;
type OwnedVec<T> = ndarray::Array<T, ndarray::Ix>;

/// Compute the discrete laplacian operator on a grid, assuming the
/// step size is 1.
/// We assume this operator operates on the C-order flattened version of
/// the grid.
///
/// This example shows how a relatively straightforward sparse matrix
/// can be constructed with a minimal number of allocations by directly
/// building up its sparse structure.
fn grid_laplacian(shape: (usize, usize)) -> sprs::CsMatOwned<f64> {
    let (rows, cols) = shape;
    let nb_vert = rows * cols;
    let mut indptr = Vec::with_capacity(nb_vert + 1);
    let nnz = 5*nb_vert + 5;
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    let mut cumsum = 0;

    for i in 0..rows {
        let top_row = i == 0;
        let bottom_row = i + 1 == rows;
        let border_row = top_row || bottom_row;

        for j in 0..cols {
            indptr.push(cumsum);

            let mut add_elt = |i, j, x| {
                indices.push(i * rows + j);
                data.push(x);
                cumsum += 1;
            };

            let left_col = j == 0;
            let right_col = j + 1 == rows;
            let border_col = left_col || right_col;
            let border = border_row || border_col;
            let corner = border_row && border_col;
            if border && !corner {
                // establish Neumann boundary conditions
                // no constraint on corners
                if bottom_row {
                    add_elt(i - 1, j, -1.);
                }
                if right_col {
                    add_elt(i, j - 1, -1.);
                }
                add_elt(i, j, 1.);
                if left_col {
                    add_elt(i, j + 1, -1.);
                }
                if top_row {
                    add_elt(i + 1, j, -1.);
                }
            }
            else if !corner {
                add_elt(i - 1, j, 1.);
                add_elt(i, j - 1, 1.);
                add_elt(i, j, -4.);
                add_elt(i, j + 1, 1.);
                add_elt(i + 1, j, 1.);
            }
        }
    }

    indptr.push(cumsum);

    sprs::CsMatOwned::new((nb_vert, nb_vert), indptr, indices, data)
}

fn set_boundary_condition<F>(mut x: VecViewMut<f64>,
                          grid_shape: (usize, usize),
                          f: F
                         )
where F: Fn(usize, usize) -> f64
{
    let (rows, cols) = grid_shape;
    for i in 0..rows {
        for j in 0..cols {
            let index = i*rows + j;
            x[[index]] = f(i, j);
        }
    }
}

fn main() {
    let lap = grid_laplacian((10, 10));
    let mut x : OwnedVec<f64> = OwnedVec::zeros(100);
    set_boundary_condition(x.view_mut(), (10, 10), |_, _| 1.);
}
