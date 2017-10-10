//! Trait to be able to know at runtime if a generic scalar is an integer, a float
//! or a complex.

use num_complex::{Complex32, Complex64};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NumKind {
    Integer,
    Float,
    Complex,
}

pub trait PrimitiveKind {
    /// Informs whether a generic primitive type contains an integer,
    /// a float or a complex
    fn num_kind() -> NumKind;
}

macro_rules! integer_prim_kind_impl {
    ($prim: ty) => (
        impl PrimitiveKind for $prim {
            fn num_kind() -> NumKind {
                NumKind::Integer
            }
        }
    )
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
    ($prim: ty) => (
        impl PrimitiveKind for $prim {
            fn num_kind() -> NumKind {
                NumKind::Float
            }
        }
    )
}

float_prim_kind_impl!(f32);
float_prim_kind_impl!(f64);

macro_rules! complex_prim_kind_impl {
    ($prim: ty) => (
        impl PrimitiveKind for $prim {
            fn num_kind() -> NumKind {
                NumKind::Complex
            }
        }
    )
}

complex_prim_kind_impl!(Complex32);
complex_prim_kind_impl!(Complex64);
