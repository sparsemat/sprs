//! This module implementations to slice a matrix along the desired dimension.
//! We're using a sealed trait to enable using ranges for an idiomatic API.

use crate::range::Range;

impl<N, I, Iptr> crate::CsMatI<N, I, Iptr>
where
    I: crate::SpIndex,
    Iptr: crate::SpIndex,
{
    /// Slice the outer dimension of the matrix according to the specified
    /// range.
    pub fn slice_outer<S>(&self, range: S) -> crate::CsMatViewI<N, I, Iptr>
    where
        S: Range,
    {
        let start = range.start().unwrap_or(0);
        let end = range.end().unwrap_or_else(|| self.outer_dims());
        if end < start {
            panic!("Invalid view");
        }
        let outer_inds_slice = self.indptr.outer_inds_slice(start, end);
        let (nrows, ncols) = match self.storage() {
            crate::CSR => ((end - start), self.ncols),
            crate::CSC => (self.nrows, (end - start)),
        };
        crate::CsMatViewI {
            nrows,
            ncols,
            storage: self.storage,
            indptr: self.indptr.middle_slice(range),
            indices: &self.indices[outer_inds_slice.clone()],
            data: &self.data[outer_inds_slice],
        }
    }
}

impl<'a, N, I, Iptr> crate::CsMatViewI<'a, N, I, Iptr>
where
    I: crate::SpIndex,
    Iptr: crate::SpIndex,
{
    /// Slice the outer dimension of the matrix according to the specified
    /// range.
    pub fn slice_outer<S>(&self, range: S) -> crate::CsMatViewI<'a, N, I, Iptr>
    where
        S: Range,
    {
        let start = range.start().unwrap_or(0);
        let end = range.end().unwrap_or_else(|| self.outer_dims());
        if end < start {
            panic!("Invalid view");
        }
        let outer_inds_slice = self.indptr.outer_inds_slice(start, end);
        let (nrows, ncols) = match self.storage() {
            crate::CSR => ((end - start), self.ncols),
            crate::CSC => (self.nrows, (end - start)),
        };
        crate::CsMatViewI {
            nrows,
            ncols,
            storage: self.storage,
            indptr: self.indptr.middle_slice(range),
            indices: &self.indices[outer_inds_slice.clone()],
            data: &self.data[outer_inds_slice],
        }
    }
}

impl<'a, N, I, Iptr> crate::CsMatViewMutI<'a, N, I, Iptr>
where
    I: crate::SpIndex,
    Iptr: crate::SpIndex,
{
    /// Slice the outer dimension of the matrix according to the specified
    /// range.
    pub fn slice_outer<S>(
        &mut self,
        range: S,
    ) -> crate::CsMatViewMutI<N, I, Iptr>
    where
        S: Range,
    {
        let start = range.start().unwrap_or(0);
        let end = range.end().unwrap_or_else(|| self.outer_dims());
        if end < start {
            panic!("Invalid view");
        }
        let outer_inds_slice = self.indptr.outer_inds_slice(start, end);
        let (nrows, ncols) = match self.storage() {
            crate::CSR => ((end - start), self.ncols),
            crate::CSC => (self.nrows, (end - start)),
        };
        crate::CsMatViewMutI {
            nrows,
            ncols,
            storage: self.storage,
            indptr: self.indptr.middle_slice(range),
            indices: &self.indices[outer_inds_slice.clone()],
            data: &mut self.data[outer_inds_slice],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::CsMat;

    #[test]
    fn slice_outer() {
        let size = 11;
        let csr: CsMat<f64> = CsMat::eye(size);
        let sliced = csr.view().slice_outer(2..7);
        let mut iter = sliced.into_iter();
        assert_eq!(iter.next().unwrap(), (&1., (0, 2)));
        assert_eq!(iter.next().unwrap(), (&1., (1, 3)));
        assert_eq!(iter.next().unwrap(), (&1., (2, 4)));
        assert_eq!(iter.next().unwrap(), (&1., (3, 5)));
        assert_eq!(iter.next().unwrap(), (&1., (4, 6)));
        assert!(iter.next().is_none());
    }
}
