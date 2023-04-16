use sprs::errors::LinalgError;
use sprs::{CsMatI, CsMatViewI, PermOwnedI, SpIndex};
use suitesparse_umfpack_sys::*;

macro_rules! umfpack_impl {
    ($int: ty,
     $Context: ident,
     $Symbolic: ident,
     $Numeric: ident,
     $symbolic: ident,
     $numeric: ident,
     $solve: ident,
     $get_numeric: ident,
     $get_symbolic: ident,
     $get_lunz: ident,
     $free_numeric: ident,
     $free_symbolic: ident,
     ) => {};
}
