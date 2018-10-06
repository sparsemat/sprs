#[macro_use]
extern crate bencher;
extern crate alga;
extern crate sprs;

use alga::general::{Additive, Inverse};
use bencher::Bencher;
use sprs::CsVec;

fn csvec_neg(bench: &mut Bencher) {
    let vector =
        CsVec::new(10000, (10..9000).collect::<Vec<_>>(), vec![-1.3; 8990]);
    bench.iter(|| -vector.clone());
}

fn csvec_additive_inverse(bench: &mut Bencher) {
    let vector =
        CsVec::new(10000, (10..9000).collect::<Vec<_>>(), vec![-1.3; 8990]);
    bench.iter(|| Inverse::<Additive>::inverse(&vector));
}

benchmark_group!(benches, csvec_neg, csvec_additive_inverse);
benchmark_main!(benches);
