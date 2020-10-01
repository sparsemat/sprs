# Raw bindings to SuiteSparse's LDL

LDL is a simple sparse Cholesky factorization algorithm. This crate exposes
the signatures exposed by this library as extern functions.

For a nicer API, the crate `sprs_suitesparse_ldl` is available.

## Features

The `ldl` part of `SuiteSparse` can be built from source by passing the `static` feature to this crate. Please observe that the license of this component is LGPL-2.1 or later.
