extern crate bincode;
extern crate sprs;

#[cfg(all(feature = "serde", feature = "serde_derive"))]
fn main() {
    use sprs::CsMat;

    let m: CsMat<f32> = CsMat::<f32>::eye(3);
    let serialized = bincode::serialize(&m.view()).unwrap();
    let deserialized = bincode::deserialize::<CsMat<f32>>(&serialized).unwrap();
    assert_eq!(m, deserialized);
}
