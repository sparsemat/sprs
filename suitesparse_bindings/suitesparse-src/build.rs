fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let root = std::env::var_os("OUT_DIR").unwrap();
    println!("cargo:root={}", root.to_string_lossy());
    if std::env::var_os("CARGO_FEATURE_CAMD").is_some() {
        cc::Build::new()
            .include("SuiteSparse/SuiteSparse_config")
            .include("SuiteSparse/CAMD/Include")
            .file("SuiteSparse/CAMD/Source/camd_1.c")
            .file("SuiteSparse/CAMD/Source/camd_2.c")
            .file("SuiteSparse/CAMD/Source/camd_aat.c")
            .file("SuiteSparse/CAMD/Source/camd_control.c")
            .file("SuiteSparse/CAMD/Source/camd_defaults.c")
            .file("SuiteSparse/CAMD/Source/camd_dump.c")
            .file("SuiteSparse/CAMD/Source/camd_global.c")
            .file("SuiteSparse/CAMD/Source/camd_info.c")
            .file("SuiteSparse/CAMD/Source/camd_order.c")
            .file("SuiteSparse/CAMD/Source/camd_postorder.c")
            .file("SuiteSparse/CAMD/Source/camd_preprocess.c")
            .file("SuiteSparse/CAMD/Source/camd_valid.c")
            .cargo_metadata(false)
            .compile("camd");
    }
    if std::env::var_os("CARGO_FEATURE_LDL").is_some() {
        // We first build ldl with LDL_LONG to make the bindings to
        // the long bits of the library
        cc::Build::new()
            .include("SuiteSparse/SuiteSparse_config")
            .include("SuiteSparse/LDL/Include")
            .file("SuiteSparse/LDL/Source/ldl.c")
            .cargo_metadata(false)
            .define("LDL_LONG", None)
            .compile("ldll");
        // We must then copy this to another location since the next
        // invocation is just a compile definition
        let mut ldl_path = std::path::PathBuf::from(root.clone());
        ldl_path.push("SuiteSparse/LDL/Source/ldl.o");
        let mut ldll_path = ldl_path.clone();
        ldll_path.set_file_name("ldll.o");
        std::fs::copy(&ldl_path, &ldll_path).unwrap();
        // And now we build ldl again (in int form), and link with the long bits
        cc::Build::new()
            .include("SuiteSparse/SuiteSparse_config")
            .include("SuiteSparse/LDL/Include")
            .file("SuiteSparse/LDL/Source/ldl.c")
            .object(&ldll_path)
            .cargo_metadata(false)
            .compile("ldl");
    }
}
