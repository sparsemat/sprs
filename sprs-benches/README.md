# sprs benches

This crate is there to benchmark algorithms in sprs, both against old versions
and versus other implementations. Currently it only benchmarks the sparse matrix
product.

The benchmark can be run against scipy, but this requires a nightly compiler,
and running the benchmarks as follows:

```
cargo +nightly run --release --features nightly
```

To enable the `dl_eigen` feature, run the following command

```
cargo build --features sprs-benches/dl_eigen
```
OR

Install the C++ Eigen package globally [link](https://gitlab.com/libeigen/eigen/-/blob/master/INSTALL?ref_type=heads)
