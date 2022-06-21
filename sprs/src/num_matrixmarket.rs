use std::fmt;
use std::fmt::Display;
use num_complex::Complex;

pub struct Displayable<T>(T);

pub trait MatrixMarketDisplay
where
    Self: Sized,
{
    fn mm_display(&self) -> Displayable<&Self> {
        Displayable(self)
    }
}

impl<T : Display> MatrixMarketDisplay for T
{
    fn mm_display(&self) -> Displayable<&Self> {
        Displayable(self)
    }
}

macro_rules! default_matrixmarket_display_impl {
    ($t: ty) => {
        impl Display for Displayable<$t>
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Displayable(it) =>
                        write!(f, "{}", it)
                }
            }
        }
        impl Display for Displayable<& $t>
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Displayable(it) =>
                        write!(f, "{}", it)
                }
            }
        }
    };
}

macro_rules! complex_matrixmarket_display_impl {
    ($t: ty) => {
        impl Display for Displayable<Complex<$t>>
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Displayable(it) =>
                        write!(f, "{} {}", it.re, it.im)
                }
            }
        }
        impl Display for Displayable<&Complex<$t>>
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Displayable(it) =>
                        write!(f, "{} {}", it.re, it.im)
                }
            }
        }
    };
}

default_matrixmarket_display_impl!(i32);
default_matrixmarket_display_impl!(f32);
default_matrixmarket_display_impl!(i64);
default_matrixmarket_display_impl!(f64);

complex_matrixmarket_display_impl!(f64);
complex_matrixmarket_display_impl!(f32);
