/// Stack implementations tuned for the graph traversal algorithms
/// encountered in sparse matrix solves/factorizations
use std::default::Default;
use std::slice;

/// A double stack of fixed capacity, growing from the left to the right
/// or conversely.
///
/// Used in sparse triangular / sparse vector solves, where it is guaranteed
/// that the two parts of the stack cannot overlap.
#[derive(Debug, Clone)]
pub struct DStack<I> {
    stacks: Vec<I>,
    left_head: Option<usize>,
    right_head: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StackVal<I> {
    Enter(I),
    Exit(I),
}

impl<I: Default> Default for StackVal<I> {
    fn default() -> StackVal<I> {
        StackVal::Enter(I::default())
    }
}

impl<I> DStack<I>
where
    I: Copy,
{
    /// Create a new double stacked suited for containing at most n elements
    pub fn with_capacity(n: usize) -> DStack<I>
    where
        I: Default,
    {
        assert!(n > 1);
        DStack {
            stacks: vec![I::default(); n],
            left_head: None,
            right_head: n,
        }
    }

    /// Capacity of the dstack
    pub fn capacity(&self) -> usize {
        self.stacks.len()
    }

    /// Test whether the left stack is empty
    pub fn is_left_empty(&self) -> bool {
        self.left_head.is_none()
    }

    /// Test whether the right stack is empty
    pub fn is_right_empty(&self) -> bool {
        self.right_head == self.capacity()
    }

    /// Push a value on the left stack
    pub fn push_left(&mut self, value: I) {
        let head = self.left_head.map_or(0, |x| x + 1);
        assert!(head < self.right_head);
        self.stacks[head] = value;
        self.left_head = Some(head);
    }

    /// Push a value on the right stack
    pub fn push_right(&mut self, value: I) {
        self.right_head -= 1;
        if let Some(left_head) = self.left_head {
            assert!(self.right_head > left_head);
        }
        self.stacks[self.right_head] = value;
    }

    /// Pop a value from the left stack
    pub fn pop_left(&mut self) -> Option<I> {
        match self.left_head {
            Some(left_head) => {
                let res = self.stacks[left_head];
                self.left_head = if left_head > 0 {
                    Some(left_head - 1)
                } else {
                    None
                };
                Some(res)
            }
            None => None,
        }
    }

    /// Pop a value from the right stack
    pub fn pop_right(&mut self) -> Option<I> {
        if self.right_head >= self.stacks.len() {
            None
        } else {
            let res = self.stacks[self.right_head];
            self.right_head += 1;
            Some(res)
        }
    }

    /// Number of right elements this double stack contains
    pub fn len_right(&self) -> usize {
        let n = self.stacks.len();
        n - self.right_head
    }

    /// Clear the right stack
    pub fn clear_right(&mut self) {
        self.right_head = self.stacks.len();
    }

    /// Clear the left stack
    pub fn clear_left(&mut self) {
        self.left_head = None;
    }

    /// Iterates along the right stack without removing items
    pub fn iter_right(&self) -> slice::Iter<I> {
        self.stacks[self.right_head..].iter()
    }

    /// Push the values of the left stack onto the right stack.
    pub fn push_left_on_right(&mut self) {
        while let Some(val) = self.pop_left() {
            self.push_right(val);
        }
    }

    /// Push the values of the right stack onto the left stack.
    pub fn push_right_on_left(&mut self) {
        while let Some(val) = self.pop_right() {
            self.push_left(val);
        }
    }
}

/// Enable extraction of stack val from iterators
pub fn extract_stack_val<I>(stack_val: &StackVal<I>) -> &I {
    match *stack_val {
        StackVal::Enter(ref i) => &i,
        StackVal::Exit(ref i) => &i,
    }
}
