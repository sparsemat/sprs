use std::path::{Path, PathBuf};

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

    if std::env::var_os("CARGO_FEATURE_UMFPACK").is_some() {
        let umfpack_dir = Path::new(&"SuiteSparse/UMFPACK");
        let sources = get_source_files(umfpack_dir.join("Source"));
        cc::Build::new()
            .define("DLONG", None)  // Use 64-bit int for addresses
            .include("SuiteSparse/SuiteSparse_config")
            .include(umfpack_dir.join("Include"))
            .files(sources)
            .include("SuiteSparse/AMD/Include")
            .cargo_metadata(false)
            .compile("umfpack");
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

/// Returns a vector of all the .c files in a given directory
fn get_source_files(dir: PathBuf) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    for entry in dir.read_dir().unwrap() {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("c") {
                sources.push(path);
            }
        }
    }
    sources
}
