
#include <Eigen/SparseCore>

extern "C" {

size_t
prod_nnz(
        size_t          a_rows,
        size_t          a_cols,
        size_t          b_cols,
        const int64_t * a_indptr,
        const int64_t * a_indices,
        const double *  a_data,
        const int64_t * b_indptr,
        const int64_t * b_indices,
        const double *  b_data
    )
{
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int64_t> SpMat;
    int64_t a_nnz = a_indptr[a_rows];
    int64_t b_nnz = b_indptr[a_cols];
    Eigen::Map<const SpMat> a(
        a_rows, a_cols, a_nnz, a_indptr, a_indices, a_data
    );
    Eigen::Map<const SpMat> b(
        a_cols, b_cols, b_nnz, b_indptr, b_indices, b_data
    );
    SpMat c = a * b;
    return c.nonZeros();
}

} // extern C
