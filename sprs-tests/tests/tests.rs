mod serde_tests {
    use sprs::*;
    #[test]
    fn valid_vectors() {
        let json_vec =
            r#"{ "dim": 100, "indices": [4, 6, 10], "data": [4, 1, 8] }"#;
        let _vec: CsVecI<u8, i32> = serde_json::from_str(&json_vec).unwrap();

        let json_vec = r#"{ "dim": 200, "indices": [4, 6, 10, 120], "data": [4, 1, 8, 1] }"#;
        let _vec: CsVecI<i8, u16> = serde_json::from_str(&json_vec).unwrap();
    }

    #[test]
    fn invalid_vectors() {
        // non-sorted indices
        let json_vec =
            r#"{ "dim": 100, "indices": [4, 6, 5], "data": [4, 1, 8] }"#;
        let e: Result<CsVecI<u8, i32>, _> = serde_json::from_str(&json_vec);
        assert!(e.is_err());

        // max(indices) > dim
        let json_vec =
            r#"{ "dim": 2, "indices": [4, 6, 8], "data": [4, 1, 8] }"#;
        let e: Result<CsVecI<u8, i32>, _> = serde_json::from_str(&json_vec);
        assert!(e.is_err());

        // indices.len != data.len
        let json_vec =
            r#"{ "dim": 100, "indices": [4, 6, 8, 10], "data": [4, 1, 8] }"#;
        let e: Result<CsVecI<u8, i32>, _> = serde_json::from_str(&json_vec);
        assert!(e.is_err());

        // indice does not fit in datatype
        let json_vec =
            r#"{ "dim": 100000, "indices": [4, 6, 32768], "data": [4, 1, 8] }"#;
        let e: Result<CsVecI<u8, i16>, _> = serde_json::from_str(&json_vec);
        assert!(e.is_err());
    }

    #[test]
    fn valid_matrices() {
        let json_mat = r#"{ "storage": "CSR", "ncols": 10, "nrows": 2, "indptr": [0, 2, 3], "indices": [4, 6, 9], "data": [4, 1, 8] }"#;
        let _mat: CsMatI<u8, i32, u16> =
            serde_json::from_str(&json_mat).unwrap();
        let _mat: CsMat<u8> = serde_json::from_str(&json_mat).unwrap();
    }

    #[test]
    fn invalid_matrices() {
        // indices not sorted
        let json_mat = r#"{ "storage": "CSR", "ncols": 10, "nrows": 2, "indptr": [0, 3, 3], "indices": [4, 9, 6], "data": [4, 1, 8] }"#;
        let mat: Result<CsMatI<u8, i32, u16>, _> =
            serde_json::from_str(&json_mat);
        assert!(mat.is_err());

        // data length != indices length
        let json_mat = r#"{ "storage": "CSR", "ncols": 10, "nrows": 2, "indptr": [0, 2, 3], "indices": [4, 9, 6], "data": [4, 1, 8, 10] }"#;
        let mat: Result<CsMatI<u8, i32, u16>, _> =
            serde_json::from_str(&json_mat);
        assert!(mat.is_err());
    }

    #[test]
    fn valid_indptr() {
        let indptr = r#"{ "storage": [0, 0, 1, 2, 2, 6] }"#;
        let indptr: IndPtr<usize> = serde_json::from_str(&indptr).unwrap();
        assert_eq!(indptr.raw_storage(), &[0, 0, 1, 2, 2, 6]);

        let indptr = r#"{ "storage": [5, 5, 8, 9] }"#;
        let indptr: IndPtr<usize> = serde_json::from_str(&indptr).unwrap();
        assert_eq!(indptr.raw_storage(), &[5, 5, 8, 9]);
    }

    #[test]
    fn invalid_indptr() {
        let indptr = r#"{ "storage": [0, 0, 1, 2, 2, 1] }"#;
        let indptr: Result<IndPtr<usize>, _> = serde_json::from_str(&indptr);
        assert!(indptr.is_err());
        let indptr = r#"{ "storage": [2, 1, 2, 2, 2, 7] }"#;
        let indptr: Result<IndPtr<usize>, _> = serde_json::from_str(&indptr);
        assert!(indptr.is_err());
        // Larger than permitted by i16
        let indptr = r#"{ "storage": [0, 32768] }"#;
        let indptr: Result<IndPtr<i16>, _> = serde_json::from_str(&indptr);
        assert!(indptr.is_err());
    }
}
