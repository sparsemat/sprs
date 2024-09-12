fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let root = std::env::var_os("OUT_DIR").unwrap();

    let mut suitesparse_config = false;
    if std::env::var_os("CARGO_FEATURE_CAMD").is_some() {
        suitesparse_config = true;
        cc::Build::new()
            .define("DLONG", None)
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
            .compile("camdl");
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
        let ldll_artifact = cc::Build::new()
            .include("SuiteSparse/SuiteSparse_config")
            .include("SuiteSparse/LDL/Include")
            .file("SuiteSparse/LDL/Source/ldl.c")
            .cargo_metadata(false)
            .define("LDL_LONG", None)
            .compile_intermediates();
        let (ldll_artifact, rest) = ldll_artifact.split_first().unwrap();
        assert!(rest.is_empty());
        // And now we build ldl again (in int form), and link with the long bits
        cc::Build::new()
            .include("SuiteSparse/SuiteSparse_config")
            .include("SuiteSparse/LDL/Include")
            .file("SuiteSparse/LDL/Source/ldl.c")
            .object(ldll_artifact)
            .cargo_metadata(false)
            .compile("ldl");
    }
    if suitesparse_config {
        cc::Build::new()
            .include("SuiteSparse/SuiteSparse_config")
            .file("SuiteSparse/SuiteSparse_config/SuiteSparse_config.c")
            .cargo_metadata(false)
            .compile("suitesparseconfig");
    }
    println!("cargo:root={}", root.to_string_lossy());
}
