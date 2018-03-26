
extern crate sprs;
extern crate sprs_suitesparse_ldl;
extern crate tempdir;
extern crate reqwest;
extern crate libflate;
extern crate tar;

use tempdir::TempDir;
use std::path::Path;
use std::fs::File;
use sprs::{CsMat};

type GenError = Box<std::error::Error>;

fn try_url<P>(mx_url: &str,
              save_dir: P,
              mat_name: &str,
              rhs_name: &str) -> Result<(), GenError>
where P: AsRef<Path>
{
    let save_dir = save_dir.as_ref();
    let mut archive_name = Path::new(mx_url);
    while let Some(stem) = archive_name.file_stem() {
        if archive_name != stem {
            archive_name = Path::new(stem);
        } else {
            break;
        }
    }
    let extracted_path = save_dir.join(archive_name);
    let probe = save_dir.join(&extracted_path).join("dl_done");
    if !probe.exists() {
        println!("probe {} does not exist, downloading {}",
                 probe.to_string_lossy(),
                 mx_url);
        tar::Archive::new(
            libflate::gzip::Decoder::new(
                reqwest::get(mx_url)?
            )?
        ).unpack(save_dir)?;
        let _ = File::create(probe)?;
    } else {
        println!("Using cached archive {}", extracted_path.to_string_lossy());
    }

    let sys_mat_path = extracted_path.join(mat_name);
    let rhs_path = extracted_path.join(rhs_name);
    let sys_mat: CsMat<f64> = sprs::io::read_matrix_market(sys_mat_path)?
        .to_csc();
    let _sys_rhs: CsMat<f64> = sprs::io::read_matrix_market(rhs_path)?
        .to_csc();

    let _ldl = sprs_suitesparse_ldl::LdlNumeric::new(sys_mat.view());

    Ok(())
}

fn main() {
    let mx_url = "https://sparse.tamu.edu/MM/Mazaheri/bundle_adj.tar.gz";
    let tmp_dir = TempDir::new("sprs-chol-tmp").unwrap();
    let _save_path = tmp_dir.path()
                            .join(Path::new(mx_url).file_name().unwrap());
    let save_dir = "/tmp/";

    let mat_name = "bundle_adj.mtx";
    let rhs_name = "bundle_adj_b.mtx";

    if let Err(error) = try_url(mx_url, save_dir, mat_name, rhs_name) {
        println!("error: {} while trying url {}", error, mx_url);
    }
}

