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

pub fn reverse_cuthill_mckee<N, I, Iptr>(
    mat: CsMatViewI<N, I, Iptr>,
) -> Ordering<I>
where
    N: PartialEq,
    I: SpIndex,
    Iptr: SpIndex,
{
    debug_assert!(is_symmetric(&mat));
    assert_eq!(mat.cols(), mat.rows());

    let nb_vertices = mat.cols();
    let degrees = mat.degrees();
    let max_neighbors = degrees.iter().max().cloned().unwrap_or(0);

    // This is the 'working data', into which new neighboring, sorted vertices are inserted,
    // the next vertex to process is popped from here.
    let mut deque = VecDeque::with_capacity(nb_vertices);

    // This are all new neighbors of the currently processed vertex, they are collected here
    // to be sorted prior to being appended to 'deque'.
    let mut neighbors = Vec::with_capacity(max_neighbors);

    // Storing which vertices have already been visited.
    let mut visited = vec![false; nb_vertices];

    // Delimeting connected components inside the permutation.
    let mut connected_parts = Vec::with_capacity(nb_vertices / 16 + 1);

    // The final permutation reducing the bandwidth of the given sparse matrix.
    // Missed optimization: Work with MaybeUninit here.
    let mut perm = vec![I::default(); nb_vertices];

    for perm_index in 0..nb_vertices {
        // Find the next index to process, choosing a new starting vertex if necessary.
        let current_vertex = deque.pop_front().unwrap_or_else(|| {
            // We found a new connected component. This number will be reverse-inverted later.
            connected_parts.push(perm_index);
            // Employ the George-Liu pseudoperipheral vertex finder to find a new starting vertex.
            find_pseudoperipheral_vertex(&visited, &degrees, &mat).unwrap()
        });

        // Write the next permutation in reverse order.
        perm[nb_vertices - perm_index - 1] = I::from_usize(current_vertex);
        visited[current_vertex.index()] = true;

        // Find, sort, and push all new neighbors of the current vertex.
        let outer = mat.outer_view(current_vertex.index()).unwrap();
        neighbors.clear();
        for &neighbor in outer.indices() {
            if !visited[neighbor.index()] {
                neighbors.push((degrees[neighbor.index()], neighbor));
                visited[neighbor.index()] = true;
            }
        }

        // Missed optimization: match small sizes explicitly, sort using sorting networks.
        // This especially makes sense if swaps are predictably compiled into cmov instructions,
        // which they aren't currently, see https://github.com/rust-lang/rust/issues/53823.
        // For more information on how to do sorting networks efficiently see https://arxiv.org/pdf/1505.01962.pdf.
        neighbors.sort_unstable_by_key(|&(deg, _)| deg);

        for (_deg, neighbor) in &neighbors {
            deque.push_back(neighbor.index());
        }
    }

    connected_parts.push(nb_vertices);

    Ordering {
        perm: PermOwnedI::new(perm),
        connected_parts: connected_parts
            .into_iter()
            .map(|i| nb_vertices - i)
            .rev()
            .collect(),
    }
}

fn find_pseudoperipheral_vertex<N, I, Iptr>(
    visited: &[bool],
    degrees: &[usize],
    mat: &CsMatViewI<N, I, Iptr>,
) -> Option<usize>
where
    N: PartialEq,
    I: SpIndex,
    Iptr: SpIndex,
{
    // Choose the next available vertex as currrent starting vertex.
    let mut current = visited
        .iter()
        .enumerate()
        .find(|(_i, &a)| !a)
        .map(|(i, _a)| i)?;

    // Isolated vertices are by definition pseudoperipheral.
    if degrees[current] == 0 {
        return Some(current);
    }

    let (mut contender, mut current_height) =
        rls_contender_and_height(current, degrees, mat);

    loop {
        let (contender_contender, contender_height) =
            rls_contender_and_height(contender, degrees, mat);

        if contender_height > current_height {
            current_height = contender_height;
            current = contender;
            contender = contender_contender;
        } else {
            return Some(current);
        }
    }
}

/// Computes the rooted level structure rooted at `root`,
/// returning the index of vertex of the last level with minimum degree, called "contender", and the height of the rls.
fn rls_contender_and_height<N, I, Iptr>(
    root: usize,
    degrees: &[usize],
    mat: &CsMatViewI<N, I, Iptr>,
) -> (usize, usize)
where
    N: PartialEq,
    I: SpIndex,
    Iptr: SpIndex,
{
    // One might wonder: "Why are we not reusing the rooted level structure (rls) we build here,
    // isn't it basically the same thing we build in the rcm again, afterwards?""
    // The answer: Yes, but, No.
    //
    // The rooted level structure here differs from the one built afterwards by its order.
    // The required order is very nasty indeed: the position of a vertex in its level depends primarily on the position of
    // its neighboring vertex in the previous level, and secondarily on its degree.
    //
    // Still, one may think: "Well, then just keep the rls around, and sort it if its root is choosen as starting vertex".
    // This is easier said than done, let's consider some strategies to do that:
    // 1. Sort it without any additionally stored information. That would require going through the entire vec again, one by one.
    //    This would erase the overhead of allocating and then deallocating the rls, but making this pass performant
    //    requires non-trivial, error-prone code. Overall, this strategies perfomance gains can at most be minimal.
    // 2. Store additional information, like the delimeters of levels, neighbouring vertex delimeters, etc.
    //    That would require doing additional work (computing and storing the information) while building the rls,
    //    and does not speed up sorting afterwards significantly, as the levels still need to be sorted serially.
    //    Overall, this strategy comes at a significant cost in memory, and it's performance improvements are debatable at best.
    // 3. Maybe just build any rls in a way that makes it a valid rcm odering?
    //    That would be optimal if we always find a pseudoperipheral vertex on first try. Unfortunately, we rarely do,
    //    typical are a few swaps, meaning this strategy, overall, comes with a loss of performance.
    //
    // So, thats why we discard the rls. One may feel free to try on his own.

    let nb_vertices = degrees.len();

    // This is ok, if we are given a valid root we can never reach an invalid vertex.
    let mut visited = vec![false; nb_vertices];

    let mut rls = Vec::with_capacity(nb_vertices);

    // Start out by pushing the root.
    visited[root] = true;
    rls.push(root);

    let mut rls_index = 0;

    // For calculating the height.
    let mut height = 0;
    let mut current_level_countdown = 1;
    let mut next_level_countup = 0;

    // The last levels len is used to compute the contender in the end.
    let mut last_level_len = 1;

    while rls_index < rls.len() {
        let parent = rls[rls_index];
        current_level_countdown -= 1;

        let outer = mat.outer_view(parent.index()).unwrap();
        for &neighbor in outer.indices() {
            if !visited[neighbor.index()] {
                visited[neighbor.index()] = true;
                next_level_countup += 1;
                rls.push(neighbor.index());
            }
        }

        if current_level_countdown == 0 {
            if next_level_countup > 0 {
                last_level_len = next_level_countup;
            }

            current_level_countdown = next_level_countup;
            next_level_countup = 0;
            height += 1;
        }

        rls_index += 1;
    }

    // Choose the contender.
    let rls_len = rls.len();
    let last_level_start_index = rls_len - last_level_len;
    let contender = rls[last_level_start_index..rls_len]
        .iter()
        .min_by_key(|i| degrees[i.index()])
        .cloned()
        .unwrap();

    // Return the node of the last level with minimal degree along with the rls's height.
    (contender, height)
}

#[cfg(test)]
mod test {
    use super::reverse_cuthill_mckee;
    use sparse::permutation::Permutation;
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
    fn reverse_cuthill_mckee_unconnected_graph_lap_components() {
        let lap_mat = unconnected_graph_lap();
        let ordering = reverse_cuthill_mckee(lap_mat.view());
        assert_eq!(&ordering.connected_parts, &[0, 3, 12],);
    }

    #[test]
    fn reverse_cuthill_mckee_unconnected_graph_lap_perm() {
        let lap_mat = unconnected_graph_lap();
        let ordering = reverse_cuthill_mckee(lap_mat.view());
        // This is just one posible permutation. Might be silently broken, e. g. through changes in unstable sorting.
        let correct_perm =
            Permutation::new(vec![7, 9, 6, 11, 10, 3, 1, 2, 5, 8, 4, 0]);
        assert_eq!(&ordering.perm.vec(), &correct_perm.vec());
    }

    #[test]
    fn reverse_cuthill_mckee_eye() {
        let mat = CsMat::<f64>::eye(3);
        let ordering = reverse_cuthill_mckee(mat.view());
        let correct_perm = Permutation::new(vec![2, 1, 0]);
        assert_eq!(&ordering.perm.vec(), &correct_perm.vec());
    }
}
