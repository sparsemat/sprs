#[cfg(feature = "dl_eigen")]
use std::fs::File;
#[cfg(feature = "dl_eigen")]
use std::path::Path;

#[cfg(feature = "dl_eigen")]
fn collect_lib<P>(
    save_dir: P,
    lib_url: &str,
    extracted_path: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: AsRef<Path>,
{
    let save_dir = save_dir.as_ref();
    let extracted_path = save_dir.join(extracted_path);
    let probe = save_dir.join(&extracted_path).join("dl_done");
    if !probe.exists() {
        println!(
            "probe {} does not exist, downloading {}",
            probe.to_string_lossy(),
            lib_url,
        );
        tar::Archive::new(libflate::gzip::Decoder::new(
            reqwest::blocking::get(lib_url)?,
        )?)
        .unpack(save_dir)?;
        let _ = File::create(probe)?;
    } else {
        println!("Using cached archive {}", extracted_path.to_string_lossy());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "dl_eigen")]
    collect_lib(
        ".",
        "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz",
        "eigen-3.3.7",
    )?;
    #[cfg(not(feature = "dl_eigen"))]
    let eigen_include_path = "/usr/include/eigen3";
    #[cfg(feature = "dl_eigen")]
    let eigen_include_path = "eigen-3.3.7";
    let res = cc::Build::new()
        .cpp(true)
        .include(eigen_include_path)
        .file("src/eigen.cpp")
        .try_compile("eigen");
    if res.is_ok() {
        println!("cargo:rustc-cfg=feature=\"eigen\"");
    } else {
        println!(
            "cargo:warning=Could not find and compile eigen, it will not \
             be benchmarked against. You can enable it by activating the \
            'dl_eigen' feature which will download and compile it."
        );
    }
    Ok(())
}
