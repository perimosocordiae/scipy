import numpy as np
import pytest
from scipy.sparse import dok_array


def test_shape_constructor():
    empty1d = dok_array((3,))
    assert empty1d.shape == (3,)
    assert np.array_equal(empty1d.toarray(), np.zeros((3,)))

    empty2d = dok_array((3, 2))
    assert empty2d.shape == (3, 2)
    assert np.array_equal(empty2d.toarray(), np.zeros((3, 2)))

    empty3d = dok_array((3, 2, 2))
    assert empty3d.shape == (3, 2, 2)
    with pytest.raises(ValueError,
                       match='Cannot densify higher-rank sparse array'):
        empty3d.toarray()


def test_dense_constructor():
    res1d = dok_array([1, 2, 3])
    assert res1d.shape == (3,)
    assert np.array_equal(res1d.toarray(), np.array([1, 2, 3]))

    res2d = dok_array([[1, 2, 3], [4, 5, 6]])
    assert res2d.shape == (2, 3)
    assert np.array_equal(res2d.toarray(), np.array([[1, 2, 3], [4, 5, 6]]))

    res3d = dok_array([[[3]], [[4]]])
    assert res3d.shape == (2, 1, 1)


def test_dense_constructor_with_shape():
    res1d = dok_array([1, 2, 3], shape=(3,))
    assert res1d.shape == (3,)
    assert np.array_equal(res1d.toarray(), np.array([1, 2, 3]))

    res2d = dok_array([[1, 2, 3], [4, 5, 6]], shape=(2, 3))
    assert res2d.shape == (2, 3)
    assert np.array_equal(res2d.toarray(), np.array([[1, 2, 3], [4, 5, 6]]))

    res3d = dok_array([[[3]], [[4]]], shape=(2, 1, 1))
    assert res3d.shape == (2, 1, 1)


def test_dense_constructor_with_inconsistent_shape():
    with pytest.raises(ValueError, match='inconsistent shapes'):
        dok_array([1, 2, 3], shape=(4,))

    with pytest.raises(ValueError, match='inconsistent shapes'):
        dok_array([1, 2, 3], shape=(3, 1))

    with pytest.raises(ValueError, match='inconsistent shapes'):
        dok_array([[1, 2, 3]], shape=(3,))


def test_1d_sparse_constructor():
    empty1d = dok_array((3,))
    res = dok_array(empty1d)
    assert res.shape == (3,)
    assert np.array_equal(res.toarray(), np.zeros((3,)))
