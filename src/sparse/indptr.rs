//! This module defines the behavior of types suitable to be used
//! as `indptr` storage in a [`CsMatBase`].
///
/// [`CsMatBase`]: type.CsMatBase.html
use crate::indexing::SpIndex;

/// Iterator type for types implementing `IndPtrStorage`. Since the language
/// does not allow `impl Iterator` in traits yet, we have to use a concrete
/// type.
pub enum IptrIter<'a, Iptr> {
    Trivial(std::slice::Windows<'a, Iptr>),
    Offset(Iptr, std::slice::Windows<'a, Iptr>),
}

impl<'a, Iptr: SpIndex> Iterator for IptrIter<'a, Iptr> {
    type Item = (Iptr, Iptr);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match self {
            IptrIter::Trivial(iter) => iter.next().map(|x| (x[0], x[1])),
            IptrIter::Offset(offset, iter) => {
                iter.next().map(|x| (x[0] + *offset, x[1] + *offset))
            }
        }
    }
}

/// Index Pointer Storage specification.
///
/// A type implementing this trait can be used in sparse matrix algorithms in
/// the CSR or CSC format.
pub trait IndPtrStorage<Iptr: SpIndex> {
    /// The length of the index pointer storage.
    fn len(&self) -> usize;

    /// Access the index pointer element at location `i`. Returns `None`
    /// if `i >= self.len()`.
    fn get(&self, i: usize) -> Option<Iptr>;

    /// Access the index pointer element at location `i`.
    ///
    /// # Panics
    ///
    /// If `i >= self.len()`
    fn at(&self, i: usize) -> Iptr {
        <Self as IndPtrStorage<Iptr>>::get(self, i).unwrap()
    }

    /// Iterate over elements in the index pointer storage
    fn iter(&self) -> IptrIter<Iptr>;
}

impl<'a, Iptr: SpIndex> IndPtrStorage<Iptr> for &'a [Iptr] {
    fn len(&self) -> usize {
        <[Iptr]>::len(self)
    }

    fn get(&self, i: usize) -> Option<Iptr> {
        <[Iptr]>::get(self, i).map(|x| *x)
    }

    /// Iterate over elements in the index pointer storage
    fn iter(&self) -> IptrIter<Iptr> {
        IptrIter::Trivial(self.windows(2))
    }
}
