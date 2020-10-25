//! This module defines the behavior of types suitable to be used
//! as `indptr` storage in a [`CsMatBase`].
//!
//! [`CsMatBase`]: type.CsMatBase.html

use crate::errors::SprsError;
use crate::indexing::SpIndex;
use std::ops::Deref;

pub struct IndPtr<Iptr, Storage>
where
    Iptr: SpIndex,
    Storage: Deref<Target = [Iptr]>,
{
    storage: Storage,
}

impl<Iptr, Storage> IndPtr<Iptr, Storage>
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
        Ok(())
    }

    pub fn new(storage: Storage) -> Result<Self, crate::errors::SprsError> {
        IndPtr::check_structure(&storage).map(|_| IndPtr::new_trusted(storage))
    }

    pub(crate) fn new_trusted(storage: Storage) -> Self {
        IndPtr { storage }
    }

    /// The length of the underlying storage
    pub fn len(&self) -> usize {
        self.storage.len()
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
    pub fn slice(&self) -> Option<&[Iptr]> {
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

    fn offset(&self) -> Iptr {
        self.storage.get(0).cloned().unwrap_or(Iptr::zero())
    }

    /// Iterate over outer dimensions, yielding start and end indices for each
    /// outer dimension.
    pub fn iter_outer(
        &self,
    ) -> impl std::iter::DoubleEndedIterator<Item = (Iptr, Iptr)>
           + std::iter::DoubleEndedIterator<Item = (Iptr, Iptr)>
           + '_ {
        let offset = self.offset();
        self.storage.windows(2).map(move |x| {
            if offset == Iptr::zero() {
                (x[0], x[1])
            } else {
                (x[0] - offset, x[1] - offset)
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
    pub fn outer_inds(&self, i: usize) -> (Iptr, Iptr) {
        assert!(i + 1 < self.storage.len());
        let offset = self.offset();
        (self.storage[i] - offset, self.storage[i + 1] - offset)
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
