extern crate bincode;
extern crate sprs;

use sprs::CsMat;

fn main() {
    let m: CsMat<f32> = CsMat::<f32>::eye(3);
    let serialized = bincode::serialize(&m.view()).unwrap();
    let deserialized = bincode::deserialize::<CsMat<f32>>(&serialized).unwrap();
    assert_eq!(m, deserialized);
}
