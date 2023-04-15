fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_STATIC").is_some() {
        let path_to_umfpack = std::env::var("DEP_SUITESPARSE_SRC_ROOT").unwrap();
        println!("cargo:rustc-link-search=native={path_to_umfpack}");
        println!("cargo:rustc-link-lib=static=umfpack");
        // println!("cargo:rustc-link-lib=static=suitesparseconfig");
    } else {
        println!("cargo:rustc-link-lib=umfpack");
    }
}
