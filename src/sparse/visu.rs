use std::fmt;

use indexing::SpIndex;
use sparse::CsMatViewI;

pub fn print_nnz_pattern<N, I, Iptr>(mat: CsMatViewI<N, I, Iptr>)
where
    N: Clone + Default,
    I: SpIndex,
    Iptr: SpIndex,
{
    print!("{}", nnz_pattern_formatter(mat));
}

pub struct NnzPatternFormatter<'a, N, I: SpIndex, Iptr: SpIndex> {
    mat: CsMatViewI<'a, N, I, Iptr>,
}

impl<'a, N, I, Iptr> fmt::Display for NnzPatternFormatter<'a, N, I, Iptr>
where
    N: Clone + Default,
    I: SpIndex,
    Iptr: SpIndex,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut write_csr = |mat: &CsMatViewI<N, I, Iptr>| {
            for row_vec in mat.outer_iterator() {
                let mut cur_col = 0;
                write!(f, "|")?;
                for (col_ind, _) in row_vec.iter() {
                    while cur_col < col_ind {
                        write!(f, " ")?;
                        cur_col += 1;
                    }
                    write!(f, "x")?;
                    cur_col = col_ind + 1;
                }
                while cur_col < mat.cols() {
                    write!(f, " ")?;
                    cur_col += 1;
                }
                write!(f, "|\n")?;
            }
            Ok(())
        };
        if self.mat.is_csr() {
            write_csr(&self.mat)
        } else {
            let mat_csr = self.mat.to_other_storage();
            write_csr(&mat_csr.view())
        }
    }
}

pub fn nnz_pattern_formatter<N, I, Iptr>(
    mat: CsMatViewI<N, I, Iptr>,
) -> NnzPatternFormatter<N, I, Iptr>
where
    I: SpIndex,
    Iptr: SpIndex,
{
    NnzPatternFormatter { mat }
}

#[cfg(test)]
mod test {
    use super::nnz_pattern_formatter;
    use sparse::CsMat;

    #[test]
    fn test_nnz_pattern_formatter() {
        let mat = CsMat::new_csc(
            (3, 3),
            vec![0, 1, 3, 4],
            vec![1, 0, 2, 2],
            vec![1.; 4],
        );
        let expected_str = "| x |\n\
                            |x  |\n\
                            | xx|\n";
        let pattern_str = format!("{}", nnz_pattern_formatter(mat.view()));
        assert_eq!(expected_str, pattern_str);
    }
}
