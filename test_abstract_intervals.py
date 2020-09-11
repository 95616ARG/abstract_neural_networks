"""Test cases for the abstraction."""
import itertools
import numpy as np
try:
    from external.bazel_python.pytest_helper import main
    IN_BAZEL = True
except ImportError:
    IN_BAZEL = False
from interval_domain import IntervalDomain, IntervalMatrix
from abstract import abstract_layer_wise, scale_columns, PCMs

def ReLU(x):
    """ReLU activation function."""
    return np.maximum(x, 0.)

def Identity(x):
    """Identity activation function."""
    return x

def test_paper_example():
    """Tests abstraction using the INN example from [1]."""
    network = []
    network.append((np.array([[1], [-1]]), ReLU))
    network.append((np.array([[1, 1], [1, 0], [0, 1]]), Identity))

    partitionings = []
    # The single input node gets its own partition.
    partitionings.append([[0]])
    # Both hidden nodes are collapsed into a single partition.
    partitionings.append([[0, 1]])
    # Each output node gets its own partition.
    partitionings.append([[0], [1], [2]])

    # Use the Interval weight set domain for both layers.
    domains = [IntervalDomain(), IntervalDomain()]

    # Abstract the neural network and get the corresponding INN.
    abstracted = abstract_layer_wise(network, partitionings, domains)

    assert abstracted[0] == (
        domains[0],
        IntervalMatrix(np.array([[-1.]]), np.array([[1.]])),
        ReLU)
    assert abstracted[1] == (
        domains[1],
        IntervalMatrix(np.array([[2.], [0.], [0.]]),
                       np.array([[2.], [2.], [2.]])),
        Identity)

def test_PCMs():
    """Unit test for the PCMs(...) method."""
    partitioning = [[0, 2], [1]]
    try:
        output = list(PCMs(partitioning))
    except NotImplementedError:
        pass
    else:
        assert False
    output = list(PCMs(partitioning, only_binary=True))
    truth = [
        np.array([[1., 0.],
                  [0., 1.],
                  [0., 0.]]),
        np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.]]),
    ]
    assert len(output) == len(truth)
    for matrix in truth:
        assert any(
            np.array_equal(out_matrix, matrix)
            for out_matrix in output)

def test_scale_columns():
    """Unit test for the scale_columns(...) method."""
    for _ in range(10):
        n, m = np.random.randint(10, 500, 2)
        matrix = np.random.uniform(size=(n, m))
        scales = list(np.random.randint(0, 500, m))
        scaled = scale_columns(matrix, scales)
        for i, j in itertools.product(range(n), range(m)):
            assert scaled[i][j] == matrix[i][j] * scales[j]

def test_interval_str():
    """Tests that str(IntervalMatrix) works correctly."""
    matrix = IntervalMatrix(np.array([[1., 2.], [3., 4], [5., 6.]]),
                            np.array([[7., 8.], [9., 10.], [11., 12.]]))
    assert (str(matrix) ==
            "[[1.0, 7.0], [2.0, 8.0]\n" +
            " [3.0, 9.0], [4.0, 10.0]\n" +
            " [5.0, 11.0], [6.0, 12.0]]")

if IN_BAZEL:
    main(__name__, __file__)
