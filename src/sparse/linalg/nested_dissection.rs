use std::collections::vec_deque::VecDeque;

use indexing::SpIndex;

use sparse::permutation::PermOwnedI;
use sparse::CsMatViewI;

/// Compute the nested dissection of a sparse matrix.
///
/// A nested dissection is a permutation that induces a block-structure and
/// a balanced optimization tree, especially for sparse matrices linked to the
/// connectivity on manifolds (such as triangle meshes).
///
/// The permutation is obtained by recursively splitting the connectivity graph
/// in two parts containing roughly the same number of vertices, separated by
/// a small interface. The returned permutation sorts the vertices such that
/// the vertices connected in the same part belong together.
pub fn nested_dissection<N, I, Iptr>(
    mat: CsMatViewI<N, I, Iptr>,
    block_size: usize,
) -> PermOwnedI<I>
where
    I: SpIndex,
    Iptr: SpIndex,
{
    assert_eq!(mat.cols(), mat.rows());
    let nb_vertices = mat.cols();
    let mut perm: Vec<_> = (0..nb_vertices).map(SpIndex::from_usize).collect();
    let max_neighbors = mat
        .indptr()
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or(Iptr::zero());
    let mut deque = VecDeque::with_capacity(max_neighbors.index());
    let mut new_perm = Vec::with_capacity(nb_vertices);
    let mut status = vec![VertStatus::Unvisited; nb_vertices];
    // need to prevent BFS from going outside the current region
    let mut in_region = vec![false; nb_vertices];
    split_connected_components_rec(
        &mat,
        &mut perm[..],
        block_size,
        &mut deque,
        &mut new_perm,
        &mut status[..],
        &mut in_region[..],
    );
    PermOwnedI::new(perm)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum VertStatus {
    Unvisited,
    FirstPart,
    Border,
    SecondPart,
}

/// Compute nested dissection of the permuted submatrix delimited by the
/// `start` and `stop` vertices.
fn nested_dissection_rec<N, I, Iptr>(
    mat: &CsMatViewI<N, I, Iptr>,
    perm: &mut [I],
    block_size: usize,
    deque: &mut VecDeque<I>,
    new_perm: &mut Vec<I>,
    status: &mut [VertStatus],
    in_region: &mut [bool],
) where
    I: SpIndex,
    Iptr: SpIndex,
{
    let nb_vertices = mat.cols();
    assert_eq!(nb_vertices, mat.rows());
    assert_eq!(nb_vertices, new_perm.capacity());
    assert_eq!(nb_vertices, status.len());
    assert_eq!(nb_vertices, in_region.len());
    if perm.len() < block_size {
        return;
    }
    use self::VertStatus::*;
    // Find the a first subgraph with a bit less than half the vertices using
    // DFS
    new_perm.clear();
    for st in status.iter_mut() {
        *st = Unvisited;
    }
    for in_rg in in_region.iter_mut() {
        *in_rg = false;
    }
    for i in perm.iter() {
        in_region[i.index()] = true;
    }
    deque.clear();
    deque.push_back(perm[0]);
    loop {
        // Should never panic as we should always be given a connex region
        // meaning split_connected_components_rec should have been called
        // before this function
        let cur_vert = deque.pop_front().unwrap();
        if status[cur_vert.index()] == FirstPart {
            continue; // already visited
        }
        new_perm.push(cur_vert);
        status[cur_vert.index()] = FirstPart;
        let outer = mat.outer_view(cur_vert.index()).unwrap();
        for &neighbor in outer.indices() {
            if status[neighbor.index()] == Unvisited
                && in_region[neighbor.index()]
            {
                deque.push_back(neighbor);
                status[neighbor.index()] = Border;
            }
        }
        if perm.len() - new_perm.len() - deque.len() <= new_perm.len() {
            break;
        }
    }
    // The vertices left in the queue are the border vertices,
    // the rest are the second connected component.
    let second_start = new_perm.len();
    for vert in perm.iter() {
        if status[vert.index()] == Unvisited && in_region[vert.index()] {
            status[vert.index()] = SecondPart;
            new_perm.push(*vert);
        }
    }
    let second_stop = new_perm.len();
    for (vert, stat) in status.iter().enumerate() {
        if *stat == Border {
            // can't overflow here, would have overflown creating the perm
            new_perm.push(I::from_usize_unchecked(vert));
        }
    }
    // Update the perm and recurse on both parts.
    perm.copy_from_slice(new_perm);
    nested_dissection_rec(
        mat,
        &mut perm[..second_start],
        block_size,
        deque,
        new_perm,
        status,
        in_region,
    );
    nested_dissection_rec(
        mat,
        &mut perm[second_start..second_stop],
        block_size,
        deque,
        new_perm,
        status,
        in_region,
    );
}

fn split_connected_components_rec<N, I, Iptr>(
    mat: &CsMatViewI<N, I, Iptr>,
    perm: &mut [I],
    block_size: usize,
    deque: &mut VecDeque<I>,
    new_perm: &mut Vec<I>,
    status: &mut [VertStatus],
    in_region: &mut [bool],
) where
    I: SpIndex,
    Iptr: SpIndex,
{
    let nb_vertices = mat.cols();
    assert_eq!(nb_vertices, mat.rows());
    assert_eq!(nb_vertices, new_perm.capacity());
    assert_eq!(nb_vertices, status.len());
    use self::VertStatus::*;
    new_perm.clear();
    for st in status.iter_mut() {
        *st = Unvisited;
    }
    deque.clear();
    deque.push_back(perm[0]);
    loop {
        let cur_vert = match deque.pop_front() {
            None => break,
            Some(cur_vert) => cur_vert,
        };
        if status[cur_vert.index()] == FirstPart {
            continue; // already visited
        }
        new_perm.push(cur_vert);
        status[cur_vert.index()] = FirstPart;
        let outer = mat.outer_view(cur_vert.index()).unwrap();
        for &neighbor in outer.indices() {
            if status[neighbor.index()] == Unvisited {
                deque.push_back(neighbor);
            }
        }
    }
    if new_perm.len() != perm.len() {
        let rest_start = new_perm.len();
        for vert in perm.iter() {
            if status[vert.index()] == Unvisited {
                status[vert.index()] = SecondPart;
                new_perm.push(*vert);
            }
        }
        perm.copy_from_slice(new_perm);
        nested_dissection_rec(
            &mat,
            &mut perm[..rest_start],
            block_size,
            deque,
            new_perm,
            &mut status[..],
            &mut in_region[..],
        );
        split_connected_components_rec(
            mat,
            &mut perm[rest_start..],
            block_size,
            deque,
            new_perm,
            status,
            in_region,
        );
    } else {
        nested_dissection_rec(
            &mat,
            &mut perm[..],
            block_size,
            deque,
            new_perm,
            &mut status[..],
            &mut in_region[..],
        );
    }
}

#[cfg(test)]
mod test {
    use super::nested_dissection;
    use sparse::CsMat;

    #[test]
    fn simple_nested_dissection() {
        // Take the laplacian matrix of the following graph
        // (no border conditions):
        //
        // 0 - 1 - 2 - 3
        // |   |   |   |
        // 4 - 5 - 6 - 7
        // |   |   |   |
        // 8 - 9 - A - B
        //
        // The laplacian matrix structure is (with x = -1)
        //       0 1 2 3 4 5 6 7 8 9 A B
        //     | 2 x     x               | 0
        //     | x 3 x     x             | 1
        //     |   x 3 x     x           | 2
        // L = |     x 2       x         | 3
        //     | x       3 x     x       | 4
        //     |   x     x 4 x     x     | 5
        //     |     x     x 4 x     x   | 6
        //     |       x     x 3       x | 7
        //     |         x       2 x     | 8
        //     |           x     x 3 x   | 9
        //     |             x     x 3 x | A
        //     |               x     x 2 | B
        let x = -1.;
        #[rustfmt::skip]
        let lap_mat = CsMat::new(
            (12, 12),
            vec![0, 3, 7, 11, 14, 18, 23, 28, 32, 35, 39, 43, 46],
            vec![0, 1, 4,
                 0, 1, 2, 5,
                 1, 2, 3, 6,
                 2, 3, 7,
                 0, 4, 5, 8,
                 1, 4, 5, 6, 9,
                 2, 5, 6, 7, 10,
                 3, 6, 7, 11,
                 4, 8, 9,
                 5, 8, 9, 10,
                 6, 9, 10, 11,
                 7, 10, 11,],
            vec![2., x, x,
                 x, 3., x, x,
                 x, 3., x, x,
                 x, 2., x,
                 x, 3., x, x,
                 x, x, 4., x, x,
                 x, x, 4., x, x,
                 x, x, 3., x,
                 x, 2., x,
                 x, x, 3., x,
                 x, x, 3., x,
                 x, x, 2.],
        );
        // let's test a single pass of the dissection
        // Unrolled BFS:
        // queue is [0]
        // visit 0
        // queue is [1, 4]
        // visit 1
        // queue is [4, 2, 5]
        // visit 4
        // queue is [2, 5, 8]
        // visit 2
        // queue is [5, 8, 3, 6]
        // visit 5
        // queue is [8, 3, 6, 9]
        // visit 8
        // queue is [3, 6, 9]
        // Stop BFS here since we have visited 6 vertices
        // Final permutation is thus
        #[rustfmt::skip]
        let expected_perm = [
            0, 1, 4, 2,   // first component
            7, 9, 10, 11, // second component
            3, 5, 6, 8,   // border
        ];
        let perm = nested_dissection(lap_mat.view(), 7);
        assert_eq!(&expected_perm, &perm.vec()[..]);

        // let's try with a second level of dissection now
        #[rustfmt::skip]
        let expected_perm2 = [
            // first component after first pass
            0, // first component after second pass
            2, // second component after second pass
            1, 4, // border after second pass
            // second component after first pass
            7, 11, // first component after second pass
            9, // second component after second pass
            10, // border after second pass
            // border after first pass
            3, 5, 6, 8,
        ];
        let perm2 = nested_dissection(lap_mat.view(), 4);
        assert_eq!(&expected_perm2, &perm2.vec()[..]);
    }

    // Laplacian matrix on a grid is already blocky by design,
    // better to build on an "irregular" mesh (eg by permuting vertices
    // on a grid that has been triangulated).
    // Also test with a non-connex graph.
    #[test]
    fn nested_dissection_non_connex_irregular_grid() {
        // Take the laplacian matrix of the following graph
        // (no border conditions):
        //
        // 0 - 4 - 2   6
        // | \ | / |   |
        // 8 - 5 - 3   9
        // | / | \ |   |
        // 1 - A - B   7
        //
        // The laplacian matrix structure is (with x = -1)
        //       0 1 2 3 4 5 6 7 8 9 A B
        //     | 3       x x     x       | 0
        //     |   3       x     x   x   | 1
        //     |     3 x x x             | 2
        // L = |     x 3   x           x | 3
        //     | x   x   3 x             | 4
        //     | x x x x x 8     x   x x | 5
        //     |             1     x     | 6
        //     |               1   x     | 7
        //     | x x       x     3       | 8
        //     |             x x   2     | 9
        //     |   x       x         3 x | A
        //     |       x   x         x 3 | B
        let x = -1.;
        #[rustfmt::skip]
        let lap_mat = CsMat::new(
            (12, 12),
            vec![0, 4, 8, 12, 16, 20, 29, 31, 33, 37, 40, 44, 48],
            vec![0, 4, 5, 8,
                 1, 5, 8, 10,
                 2, 3, 4, 5,
                 2, 3, 5, 11,
                 0, 2, 4, 5,
                 0, 1, 2, 3, 4, 5, 8, 10, 11,
                 6, 9,
                 7, 9,
                 0, 1, 5, 8,
                 6, 7, 9,
                 1, 5, 10, 11,
                 3, 5, 10, 11],
            vec![3., x, x, x,
                 3., x, x, x,
                 3., x, x, x,
                 x, 3., x, x,
                 x, x, 3., x,
                 x, x, x, x, x, 8., x, x, x,
                 1., x,
                 1., x,
                 x, x, x, 3.,
                 x, x, 2.,
                 x, x, 3., x,
                 x, x, x, 3.],
        );
        // test we have no panic due to non-connexity
        let _perm = nested_dissection(lap_mat.view(), 5);
    }
}
