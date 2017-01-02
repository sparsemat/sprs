///! Abstraction over types of indices

use std::ops::AddAssign;
use std::fmt::Debug;

use num_traits::int::PrimInt;

// TODO: maybe some combination of Into and num traits could be more ergonomic?

/// A sparse matrix index
///
/// This is a convenience trait to enable using various integer sizes for sparse
/// matrix indices.
pub trait SpIndex: Debug + PrimInt + AddAssign<Self> + Default
{

    /// Convert to usize
    ///
    /// # Panics
    ///
    /// If the integer cannot be represented as an `usize`, eg negative numbers.
    fn index(self) -> usize;

    /// Convert from usize
    fn from_usize(ind: usize) -> Self;
}

impl SpIndex for usize {
    #[inline(always)]
    fn index(self) -> usize {
        self
    }

    #[inline(always)]
    fn from_usize(ind: usize) -> Self {
        ind
    }
}

macro_rules! sp_index_signed_impl {
    ($int: ident) => (
        impl SpIndex for $int {
            #[inline(always)]
            fn index(self) -> usize {
                debug_assert!(self >= 0);
                self as usize
            }

            #[inline(always)]
            fn from_usize(ind: usize) -> Self {
                let max = $int::max_value() as usize;
                debug_assert!(ind <= max);
                ind as $int
            }
        }
    )
}

sp_index_signed_impl!(isize);
sp_index_signed_impl!(i64);
sp_index_signed_impl!(i32);
sp_index_signed_impl!(i16);

macro_rules! sp_index_unsigned_impl {
    ($int: ident) => (
        impl SpIndex for $int {
            #[inline(always)]
            fn index(self) -> usize {
                self as usize
            }

            #[inline(always)]
            fn from_usize(ind: usize) -> Self {
                let max = $int::max_value() as usize;
                debug_assert!(ind <= max);
                ind as $int
            }
        }
    )
}

sp_index_unsigned_impl!(u64);
sp_index_unsigned_impl!(u32);
sp_index_unsigned_impl!(u16);

#[cfg(test)]
mod test {
    use super::SpIndex;

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn overflow_u16() {
        let b: u16 = u16::from_usize(131072); // 2^17
        println!("{}", b);
    }
}
