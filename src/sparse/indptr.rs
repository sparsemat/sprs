//! This module defines the behavior of types suitable to be used
//! as `indptr` storage in a [`CsMatBase`].
//!
//! [`CsMatBase`]: type.CsMatBase.html

use crate::errors::SprsError;
use crate::indexing::SpIndex;
use std::ops::Deref;
use std::ops::Range;

#[derive(Eq, PartialEq, Debug, Copy, Clone, Hash)]
pub struct IndPtrBase<Iptr, Storage>
where
    Iptr: SpIndex,
    Storage: Deref<Target = [Iptr]>,
{
    storage: Storage,
}

pub type IndPtr<Iptr> = IndPtrBase<Iptr, Vec<Iptr>>;
pub type IndPtrView<'a, Iptr> = IndPtrBase<Iptr, &'a [Iptr]>;

impl<Iptr, Storage> IndPtrBase<Iptr, Storage>
where
    Iptr: SpIndex,
    Storage: Deref<Target = [Iptr]>,
{
    pub(crate) fn check_structure(storage: &Storage) -> Result<(), SprsError> {
        for i in storage.iter() {
            if i.try_index().is_none() {
                return Err(SprsError::IllegalArguments(
                    "Indptr value out of range of usize",
                ));
            }
        }
        if !storage
            .windows(2)
            .all(|x| x[0].index_unchecked() <= x[1].index_unchecked())
        {
            return Err(SprsError::UnsortedIndptr);
        }
        if storage
            .last()
            .cloned()
            .map(Iptr::index_unchecked)
            .map(|i| i > usize::max_value() / 2)
            .unwrap_or(false)
        {
            // We do not allow indptr values to be larger than half
            // the maximum value of an usize, as that would clearly exhaust
            // all available memory
            // This means we could have an isize, but in practice it's
            // easier to work with usize for indexing.
            return Err(SprsError::IllegalArguments(
                "An indptr value is larger than allowed",
            ));
        }
        if storage.len() == 0 {
            // An empty matrix has an inptr of size 1
            return Err(SprsError::IllegalArguments(
                "An indptr should have its len >= 1",
            ));
        }
        Ok(())
    }

    pub fn new(storage: Storage) -> Result<Self, crate::errors::SprsError> {
        IndPtrBase::check_structure(&storage)
            .map(|_| IndPtrBase::new_trusted(storage))
    }

    pub(crate) fn new_trusted(storage: Storage) -> Self {
        IndPtrBase { storage }
    }

    /// The length of the underlying storage
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Tests whether this indptr is empty
    pub fn is_empty(&self) -> bool {
        // An indptr of len 0 is nonsensical, we should treat that as empty
        // but fail on debug
        debug_assert!(self.storage.len() != 0);
        self.storage.len() <= 1
    }

    /// The number of outer dimensions this indptr represents
    pub fn outer_dims(&self) -> usize {
        if self.storage.len() >= 1 {
            self.storage.len() - 1
        } else {
            0
        }
    }

    /// Indicates whether the underlying storage is proper, which means the
    /// indptr corresponds to a non-sliced matrix.
    ///
    /// An empty matrix is considered non-proper.
    pub fn is_proper(&self) -> bool {
        self.storage
            .get(0)
            .map(|i| *i == Iptr::zero())
            .unwrap_or(false)
    }

    /// Return a view on the underlying slice if it is a proper `indptr` slice,
    /// which is the case if its first element is 0. `None` will be returned
    /// otherwise.
    pub fn as_slice(&self) -> Option<&[Iptr]> {
        if self.is_proper() {
            Some(&self.storage[..])
        } else {
            None
        }
    }

    /// Return a view of the underlying storage. Should be used with care in
    /// sparse algorithms, as this won't check if the storage corresponds to a
    /// proper matrix
    pub fn raw_storage(&self) -> &[Iptr] {
        &self.storage[..]
    }

    /// Returns a proper indptr representation, cloning if we do not have
    /// a proper indptr.
    pub fn to_proper(&self) -> std::borrow::Cow<[Iptr]> {
        if self.is_proper() {
            std::borrow::Cow::Borrowed(&self.storage[..])
        } else {
            let offset = self.offset();
            let proper = self.storage.iter().map(|i| *i - offset).collect();
            std::borrow::Cow::Owned(proper)
        }
    }

    fn offset(&self) -> Iptr {
        let zero = Iptr::zero();
        self.storage.get(0).cloned().unwrap_or(zero)
    }

    /// Iterate over outer dimensions, yielding start and end indices for each
    /// outer dimension.
    pub fn iter_outer(
        &self,
    ) -> impl std::iter::DoubleEndedIterator<Item = Range<Iptr>>
           + std::iter::DoubleEndedIterator<Item = Range<Iptr>>
           + '_ {
        let offset = self.offset();
        self.storage.windows(2).map(move |x| {
            if offset == Iptr::zero() {
                x[0]..x[1]
            } else {
                (x[0] - offset)..(x[1] - offset)
            }
        })
    }

    /// Return the value of the indptr at index i. This method is intended for
    /// low-level usage only, `outer_inds` should be preferred most of the time
    pub fn index(&self, i: usize) -> Iptr {
        let offset = self.offset();
        self.storage[i] - offset
    }

    /// Get the start and end indices for the requested outer dimension
    ///
    /// # Panics
    ///
    /// If `i >= self.outer_dims()`
    pub fn outer_inds(&self, i: usize) -> Range<Iptr> {
        assert!(i + 1 < self.storage.len());
        let offset = self.offset();
        (self.storage[i] - offset)..(self.storage[i + 1] - offset)
    }

    /// The number of nonzero elements described by this indptr
    pub fn nnz(&self) -> usize {
        let offset = self.offset();
        // index_unchecked validity: structure checks ensure that the last index
        // larger than the first, and that both can be represented as an usize
        self.storage
            .last()
            .map(|i| *i - offset)
            .map(Iptr::index_unchecked)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::{IndPtr, IndPtrView};

    #[test]
    fn constructors() {
        let raw_valid = vec![0, 1, 2, 3];
        assert!(IndPtr::new(raw_valid).is_ok());
        let raw_valid = vec![0, 1, 2, 3];
        assert!(IndPtrView::new(&raw_valid).is_ok());
        // Indptr for an empty matrix
        let raw_valid = vec![0];
        assert!(IndPtrView::new(&raw_valid).is_ok());
        // Indptr for an empty matrix view
        let raw_valid = vec![1];
        assert!(IndPtrView::new(&raw_valid).is_ok());

        let raw_invalid = &[0, 2, 1];
        assert_eq!(
            IndPtrView::new(raw_invalid).unwrap_err(),
            crate::errors::SprsError::UnsortedIndptr
        );
        let raw_invalid: &[usize] = &[];
        assert!(IndPtrView::new(raw_invalid).is_err());
    }

    #[test]
    fn empty() {
        assert!(IndPtrView::new(&[0]).unwrap().is_empty());
        assert!(!IndPtrView::new(&[0, 1]).unwrap().is_empty());
        #[cfg(debug_assertions)]
        {
            #[should_panic]
            assert!(IndPtrView::new_trusted(&[0]).is_empty());
        }
        #[cfg(not(debug_assertions))]
        {
            assert!(IndPtrView::new_trusted(&[0]).is_empty());
        }
    }

    #[test]
    fn outer_dims() {
        assert_eq!(IndPtrView::new(&[0]).unwrap().outer_dims(), 0);
        assert_eq!(IndPtrView::new(&[0, 1]).unwrap().outer_dims(), 1);
        assert_eq!(IndPtrView::new(&[2, 3, 5, 7]).unwrap().outer_dims(), 3);
    }

    #[test]
    fn is_proper() {
        assert!(IndPtrView::new(&[0, 1]).unwrap().is_proper());
        assert!(!IndPtrView::new(&[1, 2]).unwrap().is_proper());
    }

    #[test]
    fn offset() {
        assert_eq!(IndPtrView::new(&[0, 1]).unwrap().offset(), 0);
        assert_eq!(IndPtrView::new(&[1, 2]).unwrap().offset(), 1);
    }

    #[test]
    fn nnz() {
        assert_eq!(IndPtrView::new(&[0, 1]).unwrap().nnz(), 1);
        assert_eq!(IndPtrView::new(&[1, 2]).unwrap().nnz(), 1);
    }

    #[test]
    fn outer_inds() {
        let iptr = IndPtrView::new(&[0, 1, 3, 8]).unwrap();
        assert_eq!(iptr.outer_inds(0), 0..1);
        assert_eq!(iptr.outer_inds(1), 1..3);
        assert_eq!(iptr.outer_inds(2), 3..8);
        let res = std::panic::catch_unwind(|| iptr.outer_inds(3));
        assert!(res.is_err());
    }

    #[test]
    fn iter_outer() {
        let iptr = IndPtrView::new(&[0, 1, 3, 8]).unwrap();
        let mut iter = iptr.iter_outer();
        assert_eq!(iter.next().unwrap(), 0..1);
        assert_eq!(iter.next().unwrap(), 1..3);
        assert_eq!(iter.next().unwrap(), 3..8);
        assert!(iter.next().is_none());
    }
}
