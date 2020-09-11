"""Python code corresponding to our paper [1].

[1] Sotoudeh, M. and Thakur, A. V. Abstract Neural Networks. 27th Static
Analysis Symposium (SAS 2020).
"""
import itertools
import numpy as np

def abstract_layer_wise(network, partitionings, abstract_domains):
    """Returns an ANN corresponding to @network with given partioning, domains.

    Code corresponding to the AbstractLayerWise algorithm, Algorithm 3 in [1].

    Arguments
    =========
    @network should be a list of two-tuples (weight_matrix, activation_fn) with
        weight_matrix a Numpy array of shape (out_dims, in_dims) and
        activation_function a Python function. This matches the definition of a
        DNN in Section 3 of [1].
    @partitionings should be a list of lists of lists, where:
        1. len(@partitionings) = # layers of nodes = len(network) + 1
        2. len(@partitionings[i]) = # partitions of layer i
        3. len(@partitionings[i][j]) = # nodes in partition j of layer i
    @abstract_domains should be a list of abstract domains, with
        len(network) == len(abstract_domains). Each abstract domain must have
        (i) a domain.alpha method, which maps lists of matrices to abstract
        elements, and (ii) a boolean domain.is_convex property which indicates
        if the domain is convex.

    Return Value
    ============
    This method returns an ANN, i.e., a list of three-tuples:
        [(abstract_domain, abstract_matrix, activation_function), ...]
    with the same length as the input @network. This matches the definition of
    an ANN in Section 4 of [1]. The ith abstract domain is
    @abstract_domains[i], while the ith abstract matrix corresponds to input
    size len(@partitionings[i]) and output size len(@partitionings[i+1]).
    Finally, the ith activation function is the same as the ith activation
    function in @network.

    If all conditions in Theorem 2 of [1] are met, then the returned ANN will
    over-approximate @network.
    """
    abstract_network = []
    iterate = zip(network, partitionings, partitionings[1:], abstract_domains)
    for layer, partitions_from, partitions_to, domain in iterate:
        abstract_weights = alpha_hat(
            layer[0], partitions_from, partitions_to, domain)
        abstract_network.append((domain, abstract_weights, layer[1]))
    return abstract_network

def alpha_hat(matrix, partitioning_in, partitioning_out, abstract_domain):
    """Computes the abstraction of all mergings of @matrix.

    See Algorithm 1 in [1]. If @abstract_domain is convex, then this method
    uses the computable Algorithm 2 in [1], which is guaranteed to have the
    same result according to Theorem 1 in [1].
    """
    column_weights = list(map(len, partitioning_in))

    merged_matrices = []

    in_PCMs = PCMs(partitioning_in, only_binary=abstract_domain.is_convex)
    for in_PCM in in_PCMs:
        out_PCMs = PCMs(partitioning_out,
                        only_binary=abstract_domain.is_convex)
        for out_PCM in out_PCMs:
            merged_matrices.append(
                scale_columns(out_PCM.T @ matrix @ in_PCM, column_weights))
    return abstract_domain.alpha(merged_matrices)

def scale_columns(matrix, scales):
    """Computes a column-scaled version of @matrix with given scales.

    See Definition 17 in [1].
    """
    return np.einsum("ij,j->ij", matrix, scales)

def PCMs(partitioning, only_binary=False):
    """Yields the partitioning combination matrices for @partitioning.

    If @only_binary=True, yields only the binary PCMs, i.e., acts like BinPCMs
    in [1].
    """
    if only_binary:
        dimensions = 1 + max(
            n for partition in partitioning for n in partition)
        for assignment in itertools.product(*partitioning):
            PCM = np.zeros((dimensions, len(partitioning)))
            for partition, partition_assignment in enumerate(assignment):
                PCM[partition_assignment, partition] = 1.
            yield PCM
    else:
        # There are uncountably many non-binary PCMs.
        raise NotImplementedError
