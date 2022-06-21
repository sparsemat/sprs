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
/*
impl<T : Display> Display for Displayable<T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Displayable(it) =>
                write!(f, "{}", it)
        }
    }
}
*/

// string conversions
impl Display for Displayable<Complex<f64>>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Displayable(it) =>
                write!(f, "{} {}", it.re, it.im)
        }
    }
}

// string conversions
impl Display for Displayable<Complex<f32>>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Displayable(it) =>
                write!(f, "{} {}", it.re, it.im)
        }
    }
}

// string conversions
impl Display for Displayable<f64>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Displayable(it) =>
                write!(f, "{}", it)
        }
    }
}

// string conversions
impl Display for Displayable<f32>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Displayable(it) =>
                write!(f, "{}", it)
        }
    }
}

// string conversions
impl Display for Displayable<i64>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Displayable(it) =>
                write!(f, "{}", it)
        }
    }
}

// string conversions
impl Display for Displayable<i32>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Displayable(it) =>
                write!(f, "{}", it)
        }
    }
}
