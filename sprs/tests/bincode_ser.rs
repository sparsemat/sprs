#[cfg(feature = "serde")]
#[test]
fn serialize() {
    let bincode_config = bincode::config::standard();
    use sprs::{CsMat, CsVecI};
    let v = CsVecI::new(5, vec![0_i32, 2, 4], vec![1., 2., 3.]);
    let serialized =
        bincode::serde::encode_to_vec(&v.view(), bincode_config).unwrap();
    let deserialized =
        bincode::serde::decode_from_slice(&serialized, bincode_config).unwrap();
    assert_eq!(v, deserialized.0);

    let m: CsMat<f32> = CsMat::<f32>::eye(3);
    let serialized =
        bincode::serde::encode_to_vec(&m.view(), bincode_config).unwrap();
    let deserialized = bincode::serde::decode_from_slice::<CsMat<f32>, _>(
        &serialized,
        bincode_config,
    )
    .unwrap();
    assert_eq!(m, deserialized.0);
}
