use sprs::errors::LinalgError;
use sprs::{CsMatI, CsMatViewI, CsStructureViewI, PermOwnedI, SpIndex};
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

umfpack_impl!(
    SuiteSparseInt,
    UmfpackDIContext,
    UmfpackDISymbolic,
    UmfpackDINumeric,
    umfpack_di_symbolic_wrapper,
    umfpack_di_numeric_wrapper,
    umfpack_di_solve_wrapper,
    umfpack_di_get_numeric_wrapper,
    umfpack_di_get_symbolic_wrapper,
    umfpack_di_get_lunz,
    umfpack_di_free_numeric,
    umfpack_di_free_symbolic,
);

umfpack_impl!(
    SuiteSparseLong,
    UmfpackDLContext,
    UmfpackDLSymbolic,
    UmfpackDLNumeric,
    umfpack_dl_symbolic_wrapper,
    umfpack_dl_numeric_wrapper,
    umfpack_dl_solve_wrapper,
    umfpack_dl_get_numeric_wrapper,
    umfpack_dl_get_symbolic_wrapper,
    umfpack_dl_get_lunz,
    umfpack_dl_free_numeric,
    umfpack_dl_free_symbolic,
);
