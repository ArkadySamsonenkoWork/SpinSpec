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


def get_transformation_coeffs(basis_old: torch.Tensor, basis_new: torch.Tensor):
    """
    Calculate the squared absolute values of transformation coefficients between two bases.

    This function computes the overlap probabilities between states in two different bases.
    The output values represent |⟨basis_2_i|basis_1_j⟩|², which are the squared magnitudes
    of probability amplitudes in quantum mechanics.

    Parameters:
    -----------
    basis_1: torch.Tensor
        The first basis tensor with shape [..., K, K], where K is the spin dimension size.
        Each column basis_1[:,j] represents an eigenvector in the first basis.

    basis_2: torch.Tensor
        The second basis tensor with shape [..., K, K], where K is the spin dimension size.
        Each column basis_2[:,i] represents an eigenvector in the second basis.

    Returns:
    --------
    torch.Tensor
        A tensor of shape [..., K, K] containing the squared absolute values of the
        transformation coefficients between the two bases.

        For a 2×2 case, the output can be visualized as:

        ```
        ┌───────────────────────────────────────────┐
        │                                           │
        │     basis_1 states →                      │
        │    ┌─────────────┬─────────────┐          │
        │    │             │             │          │
        │    │  |⟨b2₀|b1₀⟩|²  |⟨b2₀|b1₁⟩|²  │        │
        │ b  │             │             │          │
        │ a  │             │             │          │
        │ s  │  |⟨b2₁|b1₀⟩|²  |⟨b2₁|b1₁⟩|²  │        │
        │ i  │             │             │          │
        │ s  │             │             │          │
        │ _  └─────────────┴─────────────┘          │
        │ 2                                         │
        │                                           │
        │ s  Element [i,j] represents the probability│
        │ t  of measuring state j in basis_1 if the │
        │ a  system was in state i in basis_2       │
        │ t                                         │
        │ e                                         │
        │ s                                         │
        │ ↓                                         │
        └───────────────────────────────────────────┘
    """

    transforms = basis_transformation(basis_old, basis_new)
    return transforms.abs().square()


def transform_rates_matrix(initial_rates, coeffs):
    """
    Transform rates given in matrix form to new basis set.
    K (b_new_1 - > b_new_2) = |⟨b_new_1|b_old_1⟩|² * |⟨b_new_2|b_old_2⟩|² * K(b_old_1 - > b_old_2)
    """
    return coeffs @ initial_rates @ coeffs.transpose(-1, -2)


def transform_rates_vector(initial_rates, coeffs):
    """
    Transform rates given in vector form to new basis set.
    K (b_new_1) = |⟨b_new_1|b_old_1⟩|² * K (b_old_1)
    """
    return torch.matmul(coeffs, initial_rates)
