from dataclasses import dataclass, field

import torch

import constants


def apply_rotations(R: torch.Tensor, T: torch.Tensor):
    """
    Rotate tensor T with respect to rotation matrices R according formula T' = RTR'
    :param R: the rotation matrices. R' = R. The shape is [rotation_dim, 3, 3]
    :param T: tensor that must be rotated. The shape is [... 3, 3]
    :return: The rotated tensors with the shape [..., rotation_dim, 3, 3]
    """
    return torch.einsum('rij, ...jk, rlk -> ...ril', R, T, R.transpose(-1, -2))

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

