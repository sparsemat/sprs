/// Stack implementations tuned for the graph traversal algorithms
/// encountered in sparse matrix solves/factorizations

use std::default::Default;

/// A double stack of fixed capacity, holding recursion information (eg for dfs)
/// as well as data values.
///
/// Used in sparse triangular / sparse vector solves, where it is guaranteed
/// that the two parts of the stack cannot overlap.
pub struct DStack<I> {
    stacks: Vec<StackVal<I>>,
    rec_head: Option<usize>,
    out_head: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StackVal<I> {
    Enter(I),
    Exit(I)
}

impl<I> DStack<I> where I: Copy {

    /// Create a new double stacked suited for containing at most n elements
    pub fn with_capacity(n: usize) -> DStack<I> where I: Default {
        assert!(n > 1);
        DStack {
            stacks: vec![StackVal::Enter(I::default()); n],
            rec_head: None,
            out_head: n
        }
    }

    /// Push a value on the recursion stack
    pub fn push_rec(&mut self, value: StackVal<I>) {
        let head = self.rec_head.map_or(0, |x| x + 1);
        assert!(head < self.out_head);
        self.stacks[head] = value;
        self.rec_head = Some(head);
    }

    /// Push a value on the data stack
    pub fn push_data(&mut self, value: I) {
        self.out_head -= 1;
        if let Some(rec_head) = self.rec_head {
            assert!(self.out_head > rec_head);
        }
        self.stacks[self.out_head] = StackVal::Enter(value);
    }

    /// Pop a value from the recursion stack
    pub fn pop_rec(&mut self) -> Option<StackVal<I>> {
        match self.rec_head {
            Some(rec_head) => {
                let res = self.stacks[rec_head];
                self.rec_head = if rec_head > 0 {
                    Some(rec_head - 1)
                } else { None };
                Some(res)
            },
            None => None
        }
    }

    /// Pop a value from the data stack
    pub fn pop_data(&mut self) -> Option<I> {
        if self.out_head >= self.stacks.len() {
            None
        }
        else {
            if let StackVal::Enter(res) = self.stacks[self.out_head] {
                self.out_head += 1;
                Some(res)
            }
            else {
                unreachable!();
            }
        }
    }

    /// Number of data elements this double stack contains
    pub fn len_data(&self) -> usize {
        let n = self.stacks.len();
        n - self.out_head
    }

    /// Clear the data stack
    pub fn clear_data(&mut self) {
        self.out_head = self.stacks.len();
    }
}
