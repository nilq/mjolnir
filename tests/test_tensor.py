import numpy as np

from mjolnir.tensor import Tensor

a = Tensor([[1, 2, 3], [4, 5, 6]])
ones = Tensor.ones(a.shape)


def test_tensor_addition():
    result = (a + ones).tensor
    expected = np.array([[2, 3, 4], [5, 6, 7]])

    assert result.shape == expected.shape, "Shape is loco."
    assert result.content.tolist() == expected.tolist(), "Content is not the same."


def test_tensor_product():
    result = (a * (ones + ones)).tensor
    expected = np.array([[2, 4, 6], [8, 10, 12]])

    assert result.shape == expected.shape, "Shape is loco."
    assert result.content.tolist() == expected.tolist(), "Content is not the same."


def test_stress():
    result = ((a + a) / ((ones + ones + ones) * (a - ones))).tensor
    assert result.shape == a.shape, "Shape is loco."
