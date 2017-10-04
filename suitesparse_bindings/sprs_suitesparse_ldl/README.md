# sprs bindings to SuiteSparse's LDL

LDL is a simple sparse Cholesky factorization algorithm. This crate provides
an interface to the original LDL implementation for users of the `sprs` crate.

Please note that a pure rust reimplementation of LDL is available in the
`sprs-ldl` crate, but it has not been as extensively tested as the original LDL
implementation.
