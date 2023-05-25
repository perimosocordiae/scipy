""" A sparse matrix in COOrdinate or 'triplet' format"""

__docformat__ = "restructuredtext en"

__all__ = ['coo_array', 'coo_matrix', 'isspmatrix_coo']

from warnings import warn

import numpy as np

from ._matrix import spmatrix
from ._sparsetools import coo_tocsr, coo_todense, coo_matvec
from ._base import isspmatrix, SparseEfficiencyWarning, _sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast, upcast_char, to_native, isshape, getdtype,
                       getdata, downcast_intp_index,
                       check_shape, check_reshape_kwargs)

import operator


class coo_array(_data_matrix, _minmax_mixin):
    """
    A sparse array in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_array(D)
            with a dense matrix D

        coo_array(S)
            with another sparse array S (equivalent to S.tocoo())

        coo_array((M, N), [dtype])
            to construct an empty sparse array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        coo_array((data, (i, j)), [shape=(M, N)])
            to construct from three arrays:
                1. data[:]   the entries of the sparse array, in any order
                2. i[:]      the row indices of the sparse array entries
                3. j[:]      the column indices of the sparse array entries

            Where ``A[i[k], j[k]] = data[k]``.  When shape is not
            specified, it is inferred from the index arrays

    Attributes
    ----------
    dtype : dtype
        Data type of the sparse array
    shape : tuple of integers
        Shape of the sparse array
    ndim : int
        Number of dimensions of the sparse array
    nnz
        Number of stored values, including explicit zeros
    data
        COO format data array of the sparse array
    row
        COO format row indices of the sparse array
    col
        COO format column indices of the sparse array

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse arrays
        - Once a COO array has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Examples
    --------

    >>> # Constructing an empty sparse array
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> coo_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a sparse array using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_array((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a sparse array with duplicate indices
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_array((data, (row, col)), shape=(4, 4))
    >>> # Duplicate indices are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    """
    _format = 'coo'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        _data_matrix.__init__(self)

        if isinstance(arg1, tuple):
            if isshape(arg1, allow_ndim=self._is_array):
                self._shape = check_shape(arg1, allow_ndim=self._is_array)
                idx_dtype = self._get_index_dtype(maxval=max(self._shape))
                data_dtype = getdtype(dtype, default=float)
                self._indices = tuple(np.array([], dtype=idx_dtype)
                                      for _ in range(len(self._shape)))
                self.data = np.array([], dtype=data_dtype)
                self.has_canonical_format = True
            else:
                try:
                    obj, indices = arg1
                except (TypeError, ValueError) as e:
                    raise TypeError('invalid input format') from e

                if shape is None:
                    if any(len(idx) == 0 for idx in indices):
                        raise ValueError('cannot infer dimensions from zero '
                                         'sized index arrays')
                    shape = tuple(operator.index(np.max(idx)) + 1
                                  for idx in indices)
                self._shape = check_shape(shape, allow_ndim=self._is_array)

                idx_dtype = self._get_index_dtype(indices,
                                                  maxval=max(self.shape),
                                                  check_contents=True)
                self._indices = tuple(np.array(idx, copy=copy, dtype=idx_dtype)
                                      for idx in indices)
                self.data = getdata(obj, copy=copy, dtype=dtype)
                self.has_canonical_format = False
        else:
            if isspmatrix(arg1):
                if isspmatrix_coo(arg1) and copy:
                    self._indices = (idx.copy() for idx in arg1._indices)
                    self.data = arg1.data.copy()
                    self._shape = check_shape(arg1.shape,
                                              allow_ndim=self._is_array)
                    self.has_canonical_format = arg1.has_canonical_format
                else:
                    coo = arg1.tocoo()
                    self._indices = tuple(coo._indices)
                    self.data = coo.data
                    self._shape = check_shape(coo.shape,
                                              allow_ndim=self._is_array)
                    self.has_canonical_format = False
            else:
                # dense argument
                M = np.asarray(arg1)
                if not self._is_array:
                    M = np.atleast_2d(M)
                    if M.ndim != 2:
                        raise TypeError('expected dimension <= 2 array or matrix')

                self._shape = check_shape(M.shape, allow_ndim=self._is_array)
                if shape is not None:
                    if check_shape(shape, allow_ndim=self._is_array) != self._shape:
                        raise ValueError('inconsistent shapes: %s != %s' %
                                         (shape, self._shape))
                index_dtype = self._get_index_dtype(maxval=max(self._shape))
                indices = M.nonzero()
                self._indices = tuple(idx.astype(index_dtype, copy=False)
                                      for idx in indices)
                self.data = M[indices]
                self.has_canonical_format = True

        if dtype is not None:
            self.data = self.data.astype(dtype, copy=False)

        self._check()

    @property
    def row(self):
        return (
            self._indices[0]
            if self.ndim > 0
            else np.array([], dtype=self._get_index_dtype())
        )

    @property
    def col(self):
        return self._indices[1] if self.ndim > 1 else np.zeros_like(self.row)

    def reshape(self, *args, **kwargs):
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)

        # Return early if reshape is not required
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        nrows, ncols = self.shape

        if order == 'C':
            # Upcast to avoid overflows: the coo_array constructor
            # below will downcast the results to a smaller dtype, if
            # possible.
            dtype = self._get_index_dtype(maxval=(ncols * max(0, nrows - 1) + max(0, ncols - 1)))

            flat_indices = np.multiply(ncols, self.row, dtype=dtype) + self.col
            new_row, new_col = divmod(flat_indices, shape[1])
        elif order == 'F':
            dtype = self._get_index_dtype(maxval=(nrows * max(0, ncols - 1) + max(0, nrows - 1)))

            flat_indices = np.multiply(nrows, self.col, dtype=dtype) + self.row
            new_col, new_row = divmod(flat_indices, shape[0])
        else:
            raise ValueError("'order' must be 'C' or 'F'")

        # Handle copy here rather than passing on to the constructor so that no
        # copy will be made of new_row and new_col regardless
        if copy:
            new_data = self.data.copy()
        else:
            new_data = self.data

        return self.__class__((new_data, (new_row, new_col)),
                              shape=shape, copy=False)

    reshape.__doc__ = _sparray.reshape.__doc__

    def _getnnz(self, axis=None):
        if axis is None:
            nnz = len(self.data)
            if nnz != len(self.row) or nnz != len(self.col):
                raise ValueError('row, column, and data array must all be the '
                                 'same length')

            if self.data.ndim != 1 or self.row.ndim != 1 or \
                    self.col.ndim != 1:
                raise ValueError('row, column, and data arrays must be 1-D')

            return int(nnz)

        if axis < 0:
            axis += 2
        if axis == 0:
            return np.bincount(downcast_intp_index(self.col),
                               minlength=self.shape[1])
        elif axis == 1:
            return np.bincount(downcast_intp_index(self.row),
                               minlength=self.shape[0])
        else:
            raise ValueError('axis out of bounds')

    _getnnz.__doc__ = _sparray._getnnz.__doc__

    def _check(self):
        """ Checks data structure for consistency """
        if self.ndim != len(self._indices):
            raise ValueError('mismatching number of index arrays for shape; '
                             f'got {len(self._indices)}, expected {self.ndim}')

        # index arrays should have integer data types
        for i, idx in enumerate(self._indices):
            if idx.dtype.kind != 'i':
                warn(f'index array {i} has non-integer dtype ({idx.dtype.name}) ')

        idx_dtype = self._get_index_dtype(self._indices, maxval=max(self.shape))
        self._indices = tuple(np.asarray(idx, dtype=idx_dtype)
                              for idx in self._indices)
        self.data = to_native(self.data)

        if self.nnz > 0:
            for i, idx in enumerate(self._indices):
                if idx.max() >= self.shape[i]:
                    raise ValueError(f'axis {i} index {idx.max()} exceeds '
                                     f'matrix dimension {self.shape[i]}')
                if idx.min() < 0:
                    raise ValueError(f'negative axis {i} index: {idx.min()}')

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation.")

        M, N = self.shape
        return self.__class__((self.data, (self.col, self.row)),
                              shape=(N, M), copy=copy)

    transpose.__doc__ = _sparray.transpose.__doc__

    def resize(self, *shape):
        shape = check_shape(shape)
        new_M, new_N = shape
        M, N = self.shape

        if new_M < M or new_N < N:
            mask = np.logical_and(self.row < new_M, self.col < new_N)
            if not mask.all():
                self.row = self.row[mask]
                self.col = self.col[mask]
                self.data = self.data[mask]

        self._shape = shape

    resize.__doc__ = _sparray.resize.__doc__

    def toarray(self, order=None, out=None):
        B = self._process_toarray_args(order, out)
        fortran = int(B.flags.f_contiguous)
        if not fortran and not B.flags.c_contiguous:
            raise ValueError("Output array must be C or F contiguous")
        if self.ndim > 2:
            raise ValueError("Cannot densify higher-rank sparse array")
        M, N, *_ = self.shape + (1, 1)
        coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                    B.ravel('A'), fortran)
        # Note: reshape() doesn't copy here, but does return a new array (view).
        return B.reshape(self.shape)

    toarray.__doc__ = _sparray.toarray.__doc__

    def tocsc(self, copy=False):
        """Convert this matrix to Compressed Sparse Column format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_array
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_array((data, (row, col)), shape=(4, 4)).tocsc()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        if self.nnz == 0:
            return self._csc_container(self.shape, dtype=self.dtype)
        else:
            M,N = self.shape
            idx_dtype = self._get_index_dtype(
                (self.col, self.row), maxval=max(self.nnz, M)
            )
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)

            indptr = np.empty(N + 1, dtype=idx_dtype)
            indices = np.empty_like(row, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))

            coo_tocsr(N, M, self.nnz, col, row, self.data,
                      indptr, indices, data)

            x = self._csc_container((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocsr(self, copy=False):
        """Convert this matrix to Compressed Sparse Row format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_array
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_array((data, (row, col)), shape=(4, 4)).tocsr()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        if self.nnz == 0:
            return self._csr_container(self.shape, dtype=self.dtype)
        else:
            M,N = self.shape
            idx_dtype = self._get_index_dtype(
                (self.row, self.col), maxval=max(self.nnz, N)
            )
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)

            indptr = np.empty(M + 1, dtype=idx_dtype)
            indices = np.empty_like(col, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))

            coo_tocsr(M, N, self.nnz, row, col, self.data,
                      indptr, indices, data)

            x = self._csr_container((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocoo(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocoo.__doc__ = _sparray.tocoo.__doc__

    def todia(self, copy=False):
        self.sum_duplicates()
        ks = self.col - self.row  # the diagonal for each nonzero
        diags, diag_idx = np.unique(ks, return_inverse=True)

        if len(diags) > 100:
            # probably undesired, should todia() have a maxdiags parameter?
            warn("Constructing a DIA matrix with %d diagonals "
                 "is inefficient" % len(diags), SparseEfficiencyWarning)

        #initialize and fill in data array
        if self.data.size == 0:
            data = np.zeros((0, 0), dtype=self.dtype)
        else:
            data = np.zeros((len(diags), self.col.max()+1), dtype=self.dtype)
            data[diag_idx, self.col] = self.data

        return self._dia_container((data, diags), shape=self.shape)

    todia.__doc__ = _sparray.todia.__doc__

    def todok(self, copy=False):
        self.sum_duplicates()
        dok = self._dok_container((self.shape), dtype=self.dtype)
        dok._update(zip(zip(self.row,self.col),self.data))

        return dok

    todok.__doc__ = _sparray.todok.__doc__

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        diag = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
                        dtype=self.dtype)
        diag_mask = (self.row + k) == self.col

        if self.has_canonical_format:
            row = self.row[diag_mask]
            data = self.data[diag_mask]
        else:
            row, _, data = self._sum_duplicates(self.row[diag_mask],
                                                self.col[diag_mask],
                                                self.data[diag_mask])
        diag[row + min(k, 0)] = data

        return diag

    diagonal.__doc__ = _data_matrix.diagonal.__doc__

    def _setdiag(self, values, k):
        M, N = self.shape
        if values.ndim and not len(values):
            return
        idx_dtype = self.row.dtype

        # Determine which triples to keep and where to put the new ones.
        full_keep = self.col - self.row != k
        if k < 0:
            max_index = min(M+k, N)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.col >= max_index)
            new_row = np.arange(-k, -k + max_index, dtype=idx_dtype)
            new_col = np.arange(max_index, dtype=idx_dtype)
        else:
            max_index = min(M, N-k)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.row >= max_index)
            new_row = np.arange(max_index, dtype=idx_dtype)
            new_col = np.arange(k, k + max_index, dtype=idx_dtype)

        # Define the array of data consisting of the entries to be added.
        if values.ndim:
            new_data = values[:max_index]
        else:
            new_data = np.empty(max_index, dtype=self.dtype)
            new_data[:] = values

        # Update the internal structure.
        self.row = np.concatenate((self.row[keep], new_row))
        self.col = np.concatenate((self.col[keep], new_col))
        self.data = np.concatenate((self.data[keep], new_data))
        self.has_canonical_format = False

    # needed by _data_matrix
    def _with_data(self,data,copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the index arrays
        (i.e. .row and .col) are copied.
        """
        if copy:
            return self.__class__((data, (self.row.copy(), self.col.copy())),
                                   shape=self.shape, dtype=data.dtype)
        else:
            return self.__class__((data, (self.row, self.col)),
                                   shape=self.shape, dtype=data.dtype)

    def sum_duplicates(self):
        """Eliminate duplicate matrix entries by adding them together

        This is an *in place* operation
        """
        if self.has_canonical_format:
            return
        summed = self._sum_duplicates(self.row, self.col, self.data)
        self.row, self.col, self.data = summed
        self.has_canonical_format = True

    def _sum_duplicates(self, row, col, data):
        # Assumes (data, row, col) not in canonical format.
        if len(data) == 0:
            return row, col, data
        order = np.lexsort((row, col))
        row = row[order]
        col = col[order]
        data = data[order]
        unique_mask = ((row[1:] != row[:-1]) |
                       (col[1:] != col[:-1]))
        unique_mask = np.append(True, unique_mask)
        row = row[unique_mask]
        col = col[unique_mask]
        unique_inds, = np.nonzero(unique_mask)
        data = np.add.reduceat(data, unique_inds, dtype=self.dtype)
        return row, col, data

    def eliminate_zeros(self):
        """Remove zero entries from the matrix

        This is an *in place* operation
        """
        mask = self.data != 0
        self.data = self.data[mask]
        self.row = self.row[mask]
        self.col = self.col[mask]

    #######################
    # Arithmetic handlers #
    #######################

    def _add_dense(self, other):
        if other.shape != self.shape:
            raise ValueError('Incompatible shapes ({} and {})'
                             .format(self.shape, other.shape))
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        result = np.array(other, dtype=dtype, copy=True)
        fortran = int(result.flags.f_contiguous)
        M, N = self.shape
        coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                    result.ravel('A'), fortran)
        return self._container(result, copy=False)

    def _mul_vector(self, other):
        #output array
        result = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char,
                                                            other.dtype.char))
        coo_matvec(self.nnz, self.row, self.col, self.data, other, result)
        return result

    def _mul_multivector(self, other):
        result = np.zeros((other.shape[1], self.shape[0]),
                          dtype=upcast_char(self.dtype.char, other.dtype.char))
        for i, col in enumerate(other.T):
            coo_matvec(self.nnz, self.row, self.col, self.data, col, result[i])
        return result.T.view(type=type(other))


def isspmatrix_coo(x):
    """Is x of coo_array type?

    Parameters
    ----------
    x
        object to check for being a coo matrix

    Returns
    -------
    bool
        True if x is a coo matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import coo_array, isspmatrix_coo
    >>> isspmatrix_coo(coo_array([[5]]))
    True

    >>> from scipy.sparse import csr_matrix, isspmatrix_coo
    >>> isspmatrix_coo(csr_matrix([[5]]))
    False
    """
    return isinstance(x, coo_array)


class coo_matrix(spmatrix, coo_array):
    """
    A sparse matrix in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_matrix(D)
            with a dense matrix D

        coo_matrix(S)
            with another sparse matrix S (equivalent to S.tocoo())

        coo_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        coo_matrix((data, (i, j)), [shape=(M, N)])
            to construct from three arrays:
                1. data[:]   the entries of the matrix, in any order
                2. i[:]      the row indices of the matrix entries
                3. j[:]      the column indices of the matrix entries

            Where ``A[i[k], j[k]] = data[k]``.  When shape is not
            specified, it is inferred from the index arrays

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of stored values, including explicit zeros
    data
        COO format data array of the matrix
    row
        COO format row index array of the matrix
    col
        COO format column index array of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse matrices
        - Once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Examples
    --------

    >>> # Constructing an empty matrix
    >>> import numpy as np
    >>> from scipy.sparse import coo_matrix
    >>> coo_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a matrix using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a matrix with duplicate indices
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
    >>> # Duplicate indices are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    """
