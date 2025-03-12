from dataclasses import dataclass, field

import torch

import constants


def apply_expanded_rotations(R: torch.Tensor, T: torch.Tensor):
    """
    Rotate tensor T with respect to rotation matrices R using T' = R T R^T.

    Parameters:
      R (torch.Tensor): Rotation matrices of shape [r, 3, 3].
      T (torch.Tensor): Tensor to rotate, with shape [..., 3, 3].

    Returns:
      torch.Tensor: Rotated tensors of shape [..., r, 3, 3].
    """
    # Add a new dimension for the rotation dimension.
    # Now T has shape [..., 1, 3, 3]
    T = T.unsqueeze(-3)

    # Expand R to allow broadcasting:
    # R is reshaped from [r, 3, 3] to [1, r, 3, 3]
    R_exp = R.unsqueeze(0)

    # First multiply: compute R * T.
    # Broadcasting aligns the new dimension in T with the r dimension in R.
    # The resulting shape is [..., r, 3, 3]
    RT = torch.matmul(R_exp, T)

    # Compute the transpose of R on the last two dimensions.
    R_T = R_exp.transpose(-2, -1)

    # Second multiplication: compute (R * T) * R^T.
    # The final shape is [..., r, 3, 3]
    rotated_T = torch.matmul(RT, R_T)

    return rotated_T


def apply_single_rotation(R: torch.Tensor, T: torch.Tensor):
    """
    Rotate tensor T with respect to rotation matrices R using T' = R T R^T.

    Applies a single rotation matrix (or a batch of rotation matrices) to a tensor
    using the transformation T' = R T R^T.

    :param R: The rotation matrices with shape [..., 3, 3].
    :param T: The tensor to be rotated with shape [..., 3, 3].
    :return: The rotated tensors with shape [..., 3, 3].
    """
    RT = torch.matmul(R, T)
    rotated_T = torch.matmul(RT, R.transpose(-2, -1))

    return rotated_T


def __apply_expanded_rotations(R: torch.Tensor, T: torch.Tensor):
    """
    Rotate tensor T with respect to rotation matrices R according formula T' = RTR'
    :param R: the rotation matrices. The shape is [rotation_dim, 3, 3]
    :param T: tensor that must be rotated. The shape is [... 3, 3]
    :return: The rotated tensors with the shape [..., rotation_dim, 3, 3]
    """
    return torch.einsum('rij, ...jk, rlk -> ...ril', R, T, R)

def __apply_single_rotation(R: torch.Tensor, T: torch.Tensor):
    """
    Rotate tensor T with respect to rotation matrices R according formula T' = RTR'

    Applies a single rotation matrix (or a batch of rotation matrices) to a tensor
    using the transformation T' = R T R^T.

    :param R: the rotation matrices. The shape is [..., 3, 3]
    :param T: tensor that must be rotated. The shape is [... 3, 3]
    :return: The rotated tensors with the shape [..., 3, 3]
    """
    return torch.einsum('...ij, ...jk, ...lk -> ...il', R, T, R)


def calculate_deriv_max(g_tensors_el: torch.Tensor, g_factors_nuc: torch.Tensor,
                        el_numbers: torch.Tensor, nuc_numbers: torch.Tensor) -> torch.Tensor:
    """
    Calculate the maximum value of the energy derivatives with respect to magnetic field.
    It is assumed that B has direction along z-axis
    :param g_tensors_el: g-tensors of electron spins. The shape is [..., 3, 3]
    :param g_factors_nuc: g-factors of the nuclei spins. The shape is [...]
    :param el_numbers: electron spin quantum numbers
    :param nuc_numbers: nuclei spins quantum numbers
    :return: the maximum value of the energy derivatives with respect to magnetic field
    """
    electron_contrib = (constants.BOHR / constants.PLANCK) * g_tensors_el[..., :, 0].sum(dim=-1) * el_numbers
    nuclear_contrib = (constants.NUCLEAR_MAGNETRON / constants.PLANCK) * g_factors_nuc * nuc_numbers
    return nuclear_contrib + electron_contrib

