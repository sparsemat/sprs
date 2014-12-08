///! A sparse matrix in the Compressed Sparse Row format
///
/// In the CSR format, a matrix is a structure containing three vectors:
/// indptr, indices, and values
/// These vectors satisfy the relation
/// for i in [0, nrows],
/// A(i, indices[indptr[i]..indptr[i+1]]) = data[indptr[i]..indptr[i+1]]

pub trait AsBorrowed<'a, N: 'a> {
    fn as_borrowed(&'a self) -> BorrowedCSR<'a, N>;
}

pub struct BorrowedCSR<'a, N: 'a> {
    nrows : uint,
    ncols : uint,
    nnz : uint,
    indptr : &'a [uint],
    indices : &'a [uint],
    data : &'a [N]
}

pub struct CSR<N> {
    nrows : uint,
    ncols : uint,
    nnz : uint,
    indptr : Vec<uint>,
    indices : Vec<uint>,
    data : Vec<N>
}

impl<'a, N: 'a> AsBorrowed<'a, N> for CSR<N> {
    fn as_borrowed(&'a self) -> BorrowedCSR<'a, N> {
        BorrowedCSR {
            nrows: self.nrows,
            ncols: self.ncols,
            nnz: self.nnz,
            indptr: self.indptr.as_slice(),
            indices: self.indices.as_slice(),
            data: self.data.as_slice(),
        }
    }
}

/// Create a CSR matrix from its main components, checking their validity
/// Validity check is performed using check_csr_structure()
pub fn new_borrowed_csr<'a, N: Clone>(
        nrows : uint, ncols: uint,
        indptr : &'a[uint], indices : &'a[uint], data : &'a[N]
        ) -> Option<BorrowedCSR<'a, N>> {
    let m = BorrowedCSR {
        nrows : nrows,
        ncols: ncols,
        nnz : data.len(),
        indptr : indptr,
        indices : indices,
        data : data
    };
    match m.check_csr_structure() {
        None => None,
        _ => Some(m)
    }
}

pub fn new_csr<N: Clone>(
        nrows : uint, ncols: uint,
        indptr : &Vec<uint>, indices : &Vec<uint>, data : &Vec<N>
        ) -> Option<CSR<N>> {
    let m = CSR {
        nrows : nrows,
        ncols: ncols,
        nnz : data.len(),
        indptr : indptr.clone(),
        indices : indices.clone(),
        data : data.clone()
    };
    match m.check_csr_structure() {
        None => None,
        _ => Some(m)
    }
}

impl<N: Clone> CSR<N> {
    fn check_csr_structure(&self) -> Option<uint> {
        self.as_borrowed().check_csr_structure()
    }
}

impl<'a, N: 'a + Clone> BorrowedCSR<'a, N> {

    /// Check the structure of CSR components
    fn check_csr_structure(&self) -> Option<uint> {
        if self.indptr.len() != self.nrows + 1 {
            println!("CSR indptr length incorrect");
            return None;
        }
        if self.indices.len() != self.data.len() {
            println!("CSR indices/data length incorrect");
            return None;
        }
        let nnz = self.indices.len();
        if nnz != self.nnz {
            println!("CSR nnz count incorrect");
            return None;
        }
        if self.indptr.iter().max().unwrap() > &nnz {
        //if ! self.indptr.iter().all(|&x| { x < nnz }) {
            println!("CSR indptr values incoherent with nnz");
            return None;
        }
        if self.indices.iter().max().unwrap() >= &self.ncols {
            println!("CSR indices values incoherent with ncols");
        //if ! self.indices.iter().all(|&x| { x < self.ncols }) {
            return None;
        }
        let mut prev_indptr : uint = 0;
        let sorted_closure = |&x| {
            let old_prev = prev_indptr.clone();
            println!("old_prev, x: {}, {}", &old_prev, &x);
            prev_indptr = x;
            x >= old_prev
        };
        if ! self.indptr.iter().all(sorted_closure) {
            println!("CSR indptr not sorted");
            return None;
        }
        // TODO: check that the indices are sorted for each row
        Some(nnz)
    }
}


#[cfg(test)]
mod test {
    use super::{new_borrowed_csr, new_csr};

    #[test]
    fn test_new_csr_success() {
        let indptr_ok : &[uint] = &[0, 1, 2, 3];
        let indices_ok : &[uint] = &[0, 1, 2];
        let data_ok : &[f64] = &[1., 1., 1.];
        match new_borrowed_csr(3, 3, indptr_ok, indices_ok, data_ok) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
    }

    #[test]
    fn test_new_csr_fails() {
        let indptr_ok : &[uint] = &[0, 1, 2, 3];
        let indices_ok : &[uint] = &[0, 1, 2];
        let data_ok : &[f64] = &[1., 1., 1.];
        let indptr_fail1 : &[uint] = &[0, 1, 2];
        let indptr_fail2 : &[uint] = &[0, 1, 2, 4];
        let indptr_fail3 : &[uint] = &[0, 2, 1, 3];
        let indices_fail1 : &[uint] = &[0, 1];
        let indices_fail2 : &[uint] = &[0, 1, 4];
        let data_fail1 : &[f64] = &[1., 1., 1., 1.];
        let data_fail2 : &[f64] = &[1., 1.,];
        match new_borrowed_csr(3, 3, indptr_fail1, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csr(3, 3, indptr_fail2, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csr(3, 3, indptr_fail3, indices_ok, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csr(3, 3, indptr_ok, indices_fail1, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csr(3, 3, indptr_ok, indices_fail2, data_ok) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csr(3, 3, indptr_ok, indices_ok, data_fail1) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
        match new_borrowed_csr(3, 3, indptr_ok, indices_ok, data_fail2) {
            Some(_) => assert!(false),
            None => assert!(true)
        }
    }

    #[test]
    fn test_new_csr_vec_borrowed() {
        let indptr_ok = vec![0u, 1, 2, 3];
        let indices_ok = vec![0u, 1, 2];
        let data_ok : Vec<f64> = vec![1., 1., 1.];
        match new_borrowed_csr(3, 3, indptr_ok.as_slice(),
                      indices_ok.as_slice(), data_ok.as_slice()) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
    }

    #[test]
    fn test_new_csr_vec_owned() {
        let indptr_ok = vec![0u, 1, 2, 3];
        let indices_ok = vec![0u, 1, 2];
        let data_ok : Vec<f64> = vec![1., 1., 1.];
        match new_csr(3, 3, &indptr_ok, &indices_ok, &data_ok) {
            Some(_) => assert!(true),
            None => assert!(false)
        }
    }
}
