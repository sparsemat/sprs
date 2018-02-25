
extern crate sprs;
extern crate sprs_suitesparse_ldl;
extern crate tempdir;
extern crate reqwest;
extern crate libflate;
extern crate tar;

use tempdir::TempDir;
use std::path::Path;
use std::io;
use std::fs::File;

fn main() {
    let mx_url = "https://sparse.tamu.edu/MM/Mazaheri/bundle_adj.tar.gz";
    let tmp_dir = TempDir::new("sprs-chol-tmp").unwrap();
    let save_path = tmp_dir.path()
                           .join(Path::new(mx_url).file_name().unwrap());
    let save_dir = "/tmp/";

    // FIXME better handle errors
    tar::Archive::new(
        libflate::gzip::Decoder::new(
            reqwest::get(mx_url).expect("cannot parse matrix url")
        ).expect("not a valid gz file")
    ).unpack(save_dir).expect("not a valid tar file");
}

