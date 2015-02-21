/// A container for dealing with a contiguous piece of data that is either
/// owned or borrowed.
///
/// This is a frequent requirement for the storage array in linear algebra codes
/// (ie a submatrix extracted from a matrix should borrow the data but be of
/// the same type).

use std::ops::Deref;

pub enum VecSlice<'a, N: 'a> {
    Vec(Vec<N>),
    Slice(&'a[N])
}

impl<'a, N: 'a> VecSlice<'a, N> {
    pub fn from_vec(v: Vec<N>) -> VecSlice<'a,N> {
        VecSlice::Vec(v)
    }

    pub fn from_slice(s: &'a[N]) -> VecSlice<'a, N> {
        VecSlice::Slice(s)
    }
}

impl<'a, N> Deref for VecSlice<'a, N> {
    type Target = [N];

    fn deref(& self) -> &[N] {
        match self {
            &VecSlice::Vec(ref v) => &v[..],
            &VecSlice::Slice(s) => s
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
}
