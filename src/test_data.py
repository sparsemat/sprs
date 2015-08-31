"""
Sparse matrices in Scipy matching those we have in rust,
to make it easy to test our code.
"""

import numpy as np
import scipy.sparse

def mat1():
    indptr = np.array([0, 2, 4, 5, 6, 7])
    indices = np.array([2, 3, 3, 4, 2, 1, 3])
    data = np.array([3., 4., 2., 5., 5., 8., 7.])
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(5,5))

def mat1_csc():
    indptr = np.array([0, 0, 1, 3, 6, 7])
    indices = np.array([3, 0, 2, 0, 1, 4, 1])
    data = np.array([8.,  3.,  5.,  4.,  2.,  7.,  5.])
    return scipy.sparse.csc_matrix((data, indices, indptr), shape=(5,5))

def mat2():
    indptr = np.array([0,  4,  6,  6,  8, 10])
    indices = np.array([0, 1, 2, 4, 0, 3, 2, 3, 1, 2])
    data = np.array([6.,  7.,  3.,  3.,  8., 9.,  2.,  4.,  4.,  4.])
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(5,5))


def mat3():
    indptr = np.array([0, 2, 4, 5, 6, 7])
    indices = np.array([2, 3, 2, 3, 2, 1, 3])
    data = np.array([3., 4., 2., 5., 5., 8., 7.])
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(5,4))


def mat4():
    indptr = np.array([0,  4,  6,  6,  8, 10])
    indices = np.array([0, 1, 2, 4, 0, 3, 2, 3, 1, 2])
    data = np.array([6.,  7.,  3.,  3.,  8., 9.,  2.,  4.,  4.,  4.])
    return scipy.sparse.csc_matrix((data, indices, indptr), shape=(5,5))
