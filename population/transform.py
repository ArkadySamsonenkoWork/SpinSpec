import torch
import torch.nn as nn


def basis_transformation(basis_1: torch.Tensor, basis_2: torch.Tensor) -> torch.Tensor:
    """
    :param basis_1: The basis function. The shape is [..., K, K], where K is spin dimension size.
    The column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].

    :param basis_2: The basis function. The shape is [..., K, K], where K is spin dimension size.
    The column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].

    :return: A transformation matrix of shape [..., K, K] that transforms
            vectors from the `basis_1` coordinate system to the `basis_2` coordinate system.
    """
    return torch.matmul(basis_2.conj().transpose(-2, -1), basis_1)
