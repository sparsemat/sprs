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
    nested_dissection_rec(
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
        let cur_vert = deque.pop_front().expect(
            "Having no more neighbors means we should have stopped earlier?\n\
             Or maybe we have a non-connex graph, FIXME need to figure it out",
        );
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
            }
        }
        if new_perm.len() >= perm.len() / 2 {
            break;
        }
    }
    // The vertices left in the queue are the border vertices,
    // the rest are the second connected component.
    for vert in deque.iter() {
        if status[vert.index()] == Unvisited && in_region[vert.index()] {
            status[vert.index()] = Border;
        }
    }
    let second_start = new_perm.len();
    for vert in perm.iter() {
        if status[vert.index()] == Unvisited && in_region[vert.index()] {
            status[vert.index()] = SecondPart;
            new_perm.push(*vert);
        }
    }
    // Need to use the status as we can have duplicates in the deque
    for (vert, stat) in status.iter().enumerate() {
        if *stat == Border {
            // can't overflow here, would have overflown creating the perm
            new_perm.push(I::from_usize_unchecked(vert));
        }
    }
    let second_stop = new_perm.len();
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
            0, 1, 4, 2, 5, 8, // first component
            7, 10, 11,        // second component
            3, 6, 9,          // border
        ];
        let perm = nested_dissection(lap_mat.view(), 7);
        assert_eq!(&expected_perm, &perm.vec()[..]);

        // let's try with a second level of dissection now
        // We'll dissect with a block size of 4 so that only the first component
        // gets dissected again
        // Unrolled BFS:
        // queue is [0]
        // visit 0
        // queue is [1, 4]
        // visit 1
        // queue is [4, 2, 5]
        // visit 4
        // queue is [2, 5, 8]
        // Stop BFS here since we have visited 3 vertices
        // Final permutation is thus
        #[rustfmt::skip]
        let expected_perm2 = [
            0, 1, 4, // first-first component
            // empty first-second component
            2, 5, 8, // first-border
            7, 10, 11, // second component
            3, 6, 9, // border
        ];
        let perm2 = nested_dissection(lap_mat.view(), 7);
        assert_eq!(&expected_perm2, &perm2.vec()[..]);
    }
}
