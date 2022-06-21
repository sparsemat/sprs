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

// string conversions
impl MatrixMarketDisplay for Complex<f64>
{
    fn mm_display(&self) -> Displayable<&Self> {
        Displayable(self)
    }
}

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


/*

// string conversions
impl MatrixMarketDisplay for Complex<f64>
{
    fn mm_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.re, self.im)
    }
}

// string conversions
impl MatrixMarketDisplay for Complex<f32>
{
    fn mm_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.re, self.im)
    }
}

// string conversions
impl MatrixMarketDisplay for i64
{
    fn mm_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

// string conversions
impl MatrixMarketDisplay for i32
{
    fn mm_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}
*/
