fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_STATIC").is_some() {
        let path_to_camd = std::env::var("DEP_SUITESPARSE_SRC_ROOT").unwrap();
        println!("cargo:rustc-link-search=native={path_to_camd}");
        println!("cargo:rustc-link-lib=static=camd");
        println!("cargo:rustc-link-lib=static=camdl");
        println!("cargo:rustc-link-lib=static=suitesparseconfig");
    } else {
        println!("cargo:rustc-link-lib=camd");
    }
}
