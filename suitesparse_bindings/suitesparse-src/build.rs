fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:root={}", std::env::var("OUT_DIR").unwrap());
    if std::env::var_os("CARGO_FEATURE_CAMD").is_some() {
        cc::Build::new()
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
        cc::Build::new()
            .include("SuiteSparse/LDL/Include")
            .file("SuiteSparse/LDL/Source/ldl.c")
            .cargo_metadata(false)
            .compile("ldl")
    }
}
