use std::collections::vec_deque::VecDeque;

use indexing::SpIndex;

use sparse::permutation::PermOwnedI;
use sparse::symmetric::is_symmetric;
use sparse::CsMatViewI;

pub struct Ordering<I> {
    /// The computed permutation
    pub perm: PermOwnedI<I>,
    /// Indices inside the permutation delimiting connected components
    pub connected_parts: Vec<usize>,
}

pub fn cuthill_mckee<N, I, Iptr>(mat: CsMatViewI<N, I, Iptr>) -> Ordering<I>
where
    N: PartialEq,
    I: SpIndex,
    Iptr: SpIndex,
{
    debug_assert!(is_symmetric(&mat));
    assert_eq!(mat.cols(), mat.rows());
    let nb_vertices = mat.cols();
    let mut deque = VecDeque::with_capacity(nb_vertices);
    let max_neighbors = mat
        .indptr()
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or(Iptr::zero());
    let mut neighbors = Vec::with_capacity(max_neighbors.index());
    let mut perm = Vec::with_capacity(nb_vertices);
    let mut visited = vec![false; nb_vertices];
    let mut connected_parts = Vec::with_capacity(4);
    connected_parts.push(0);

    let degrees = mat.degrees();

    while perm.len() < nb_vertices {
        // find the non-visited vertex with the lowest degree
        let mut min_deg = nb_vertices;
        let mut min_deg_vert = 0;
        for (vert, vis) in visited.iter().enumerate() {
            let vert_deg = degrees[vert];
            if !vis && vert_deg <= min_deg {
                min_deg = vert_deg;
                min_deg_vert = vert;
            }
        }
        deque.clear();
        deque.push_back(min_deg_vert);

        while let Some(cur_vert) = deque.pop_front() {
            if visited[cur_vert] {
                continue;
            }
            perm.push(I::from_usize(cur_vert));
            visited[cur_vert.index()] = true;
            let outer = mat.outer_view(cur_vert.index()).unwrap();
            neighbors.clear();
            for &neighbor in outer.indices() {
                if !visited[neighbor.index()] {
                    neighbors.push((degrees[neighbor.index()], neighbor));
                }
            }
            neighbors.sort_by_key(|&(deg, _)| deg);
            for (_deg, neighbor) in &neighbors {
                deque.push_back(neighbor.index());
            }
        }
        connected_parts.push(perm.len());
    }

    Ordering {
        perm: PermOwnedI::new(perm),
        connected_parts,
    }
}

#[cfg(test)]
mod test {
    use super::cuthill_mckee;
    use sparse::CsMat;

    fn unconnected_graph_lap() -> CsMat<f64> {
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
        lap_mat
    }

    #[test]
    fn cuthill_mckee_unconnected_graph_lap() {
        let lap_mat = unconnected_graph_lap();
        let ordering = cuthill_mckee(lap_mat.view());
        assert_eq!(&ordering.connected_parts, &[0, 3, 12],);
    }
}
