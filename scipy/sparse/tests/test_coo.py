import numpy as np
import pytest
from scipy.sparse import coo_array


def test_1d_shape_constructor():
    empty1d = coo_array((3,))
    assert empty1d.shape == (3,)
    assert np.array_equal(empty1d.toarray(), np.zeros((3,)))


def test_1d_dense_constructor():
    res = coo_array([1, 2, 3])
    assert res.shape == (3,)
    assert np.array_equal(res.toarray(), np.array([1, 2, 3]))


def test_1d_dense_constructor_with_shape():
    res = coo_array([1, 2, 3], shape=(3,))
    assert res.shape == (3,)
    assert np.array_equal(res.toarray(), np.array([1, 2, 3]))


def test_1d_dense_constructor_with_inconsistent_shape():
    with pytest.raises(ValueError, match='inconsistent shapes'):
        coo_array([1, 2, 3], shape=(4,))


def test_1d_sparse_constructor():
    empty1d = coo_array((3,))
    res = coo_array(empty1d)
    assert res.shape == (3,)
    assert np.array_equal(res.toarray(), np.zeros((3,)))


def test_1d_tuple_constructor():
    res = coo_array(([9,8], ([1,2],)))
    assert res.shape == (3,)
    assert np.array_equal(res.toarray(), np.array([0, 9, 8]))


def test_1d_tuple_constructor_with_shape():
    res = coo_array(([9,8], ([1,2],)), shape=(4,))
    assert res.shape == (4,)
    assert np.array_equal(res.toarray(), np.array([0, 9, 8, 0]))
