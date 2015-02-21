/// A container for dealing with a contiguous piece of data that is either
/// owned or borrowed.
///
/// This is a frequent requirement for the storage array in linear algebra codes
/// (ie a submatrix extracted from a matrix should borrow the data but be of
/// the same type).

use std::ops::{Deref, DerefMut};

pub enum VecSlice<'a, N: 'a> {
    Vec(Vec<N>),
    Slice(&'a[N]),
    MutSlice(&'a mut[N])
}

impl<'a, N: 'a> VecSlice<'a, N> {
    pub fn from_vec(v: Vec<N>) -> VecSlice<'a,N> {
        VecSlice::Vec(v)
    }

    pub fn from_slice(s: &'a[N]) -> VecSlice<'a, N> {
        VecSlice::Slice(s)
    }

    pub fn from_mut_slice(s: &'a mut[N]) -> VecSlice<'a, N> {
        VecSlice::MutSlice(s)
    }
}

impl<'a, N> Deref for VecSlice<'a, N> {
    type Target = [N];

    fn deref(& self) -> &[N] {
        match self {
            &VecSlice::Vec(ref v) => &v[..],
            &VecSlice::Slice(s) => s,
            &VecSlice::MutSlice(ref s) => s
        }
    }
}

impl<'a, N> DerefMut for VecSlice<'a, N> {
    fn deref_mut(& mut self) -> &mut[N] {
        match self {
            & mut VecSlice::Vec(ref mut v) => v.as_mut_slice(),
            & mut VecSlice::Slice(mut s) => panic!("not mutable"), // FIXME
            & mut VecSlice::MutSlice(ref mut s) => s
        }
    }
}


#[cfg(test)]
mod test {
    use super::VecSlice;

    #[test]
    fn deref_on_index() {
        let v = vec![0, 1, 2];
        let s = [0,1,2];
        let vs_vec = VecSlice::from_vec(v);
        let vs_slice = VecSlice::from_slice(&s);

        let a = vs_vec[0];
        let b = vs_slice[0];

        assert!(a == b); // just to avoid warnings
    }

    #[test]
    fn deref_mut() {
        let v = vec![0, 1, 2];
        let mut vs_vec = VecSlice::from_vec(v);
        let mut s = [0,1,2];
        let mut vs_slice_mut = VecSlice::from_mut_slice(&mut s);

        vs_vec[0] = 1;
        assert!(vs_vec[0] == 1);
        vs_slice_mut[0] = 1;
        assert!(vs_slice_mut[0] == 1);

    }

    #[test]
    #[should_fail]
    fn deref_mut_panics_on_immutable_slice() {
        let s = [0,1,2];
        let mut vs_slice = VecSlice::from_slice(&s);
        vs_slice[0] = 1;
    }
}
