# sprs benches

This crate is there to benchmark algorithms in sprs, both against old versions
and versus other implementations. Currently it only benchmarks the sparse matrix
product.

The benchmark can be run against scipy, but this requires a nightly compiler,
and running the benchmarks as follows:

```
cargo +nightly run --release --features nightly
```
