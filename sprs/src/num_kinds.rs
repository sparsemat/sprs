//! Trait to be able to know at runtime if a generic scalar is an integer, a float
//! or a complex.

use num_complex::{Complex32, Complex64};
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};
use std::{
    fmt,
    ops::{Add, Div, Mul, Rem, Sub},
};
/// the type for Pattern data, it's special which contains no data
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Pattern {}

impl Add for Pattern {
    type Output = Pattern;
    fn add(self, _other: Pattern) -> Pattern {
        Pattern {}
    }
}
impl Zero for Pattern {
    fn zero() -> Self {
        Pattern {}
    }
    fn is_zero(&self) -> bool {
        true
    }

    fn set_zero(&mut self) {
        *self = Zero::zero();
    }
}
impl Rem for Pattern {
    type Output = Self;
    fn rem(self, _rhs: Self) -> Self {
        Pattern {}
    }
}
impl Div for Pattern {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        Pattern {}
    }
}
impl Sub for Pattern {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        Pattern {}
    }
}
impl Mul for Pattern {
    type Output = Self;

    fn mul(self, _rhs: Self) -> Self::Output {
        Pattern {}
    }
}
impl One for Pattern {
    fn one() -> Self {
        Pattern {}
    }
}
impl Num for Pattern {
    type FromStrRadixErr = ();

    fn from_str_radix(
        _str: &str,
        _radix: u32,
    ) -> Result<Self, Self::FromStrRadixErr> {
        Err(())
    }
}

impl std::ops::Neg for Pattern {
    type Output = Pattern;

    fn neg(self) -> Self::Output {
        self
    }
}

impl ToPrimitive for Pattern {
    fn to_i64(&self) -> Option<i64> {
        None
    }
    fn to_u64(&self) -> Option<u64> {
        None
    }
    fn to_f64(&self) -> Option<f64> {
        None
    }
}

impl NumCast for Pattern {
    fn from<T: num_traits::ToPrimitive>(_n: T) -> Option<Self> {
        None
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NumKind {
    Integer,
    Float,
    Complex,
    Pattern,
}

impl fmt::Display for NumKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Integer => write!(f, "integer"),
            Self::Float => write!(f, "real"),
            Self::Complex => write!(f, "complex"),
            Self::Pattern => write!(f, "pattern"),
        }
    }
}

pub trait PrimitiveKind {
    /// Informs whether a generic primitive type contains an integer,
    /// a float or a complex
    fn num_kind() -> NumKind;
}
impl PrimitiveKind for Pattern {
    fn num_kind() -> NumKind {
        NumKind::Pattern
    }
}

macro_rules! integer_prim_kind_impl {
    ($prim: ty) => {
        impl PrimitiveKind for $prim {
            fn num_kind() -> NumKind {
                NumKind::Integer
            }
        }
    };
}

integer_prim_kind_impl!(i8);
integer_prim_kind_impl!(u8);
integer_prim_kind_impl!(i16);
integer_prim_kind_impl!(u16);
integer_prim_kind_impl!(i32);
integer_prim_kind_impl!(u32);
integer_prim_kind_impl!(i64);
integer_prim_kind_impl!(u64);
integer_prim_kind_impl!(isize);
integer_prim_kind_impl!(usize);

macro_rules! float_prim_kind_impl {
    ($prim: ty) => {
        impl PrimitiveKind for $prim {
            fn num_kind() -> NumKind {
                NumKind::Float
            }
        }
    };
}

float_prim_kind_impl!(f32);
float_prim_kind_impl!(f64);

macro_rules! complex_prim_kind_impl {
    ($prim: ty) => {
        impl PrimitiveKind for $prim {
            fn num_kind() -> NumKind {
                NumKind::Complex
            }
        }
    };
}

complex_prim_kind_impl!(Complex32);
complex_prim_kind_impl!(Complex64);
