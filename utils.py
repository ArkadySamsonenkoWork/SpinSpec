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


def rotation_matrix_to_euler_angles(R: torch.Tensor, convention: str = "zyz"):
    """
    Convert a 3x3 rotation matrix to ZYZ Euler angles.
    """
    # Extract elements from rotation matrix
    r11, r12, r13 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r21, r22, r23 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r31, r32, r33 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # Calculate beta (middle rotation around Y-axis)
    beta = torch.acos(torch.clamp(r33, -1.0, 1.0))

    # Check for singularities
    sin_beta = torch.sin(beta)

    if torch.abs(sin_beta) > 1e-6:  # Non-singular case
        # Calculate alpha (first rotation around Z-axis)
        alpha = torch.atan2(r23, r13)

        # Calculate gamma (last rotation around Z-axis)
        gamma = torch.atan2(r32, -r31)
    else:
        # Singular case: beta ≈ 0 or π
        if torch.abs(beta) < 1e-6:  # beta ≈ 0
            # When beta ≈ 0, only alpha + gamma is determined
            alpha = torch.atan2(r12, r11)
            gamma = torch.tensor(0.0, dtype=R.dtype, device=R.device)
        else:  # beta ≈ π
            # When beta ≈ π, only alpha - gamma is determined
            alpha = torch.atan2(-r12, r11)
            gamma = torch.tensor(0.0, dtype=R.dtype, device=R.device)
    return torch.tensor([alpha, beta, gamma])


def euler_angles_to_matrix(angles: torch.Tensor, convention: str = "zyz"):
    """
    :param euler_angles: torch.Tensor of shape (..., 3) containing Euler angles in radians
    :param convention: str, rotation convention (default 'xyz')
                   Supported: 'zyz', 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
    :return: torch.Tensor of shape (..., 3, 3) containing rotation matrices
    """
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor(angles, dtype=torch.float32)

    if angles.dim() == 1:
        angles = angles.unsqueeze(0)

    batch_shape = angles.shape[:-1]
    angles = angles.view(-1, 3)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    cx, cy, cz = cos_angles[:, 0], cos_angles[:, 1], cos_angles[:, 2]
    sx, sy, sz = sin_angles[:, 0], sin_angles[:, 1], sin_angles[:, 2]

    if convention == 'zyz':
        # R = Rz2 * Ry * Rz1
        R = torch.zeros(angles.shape[0], 3, 3, device=angles.device, dtype=angles.dtype)
        R[:, 0, 0] = cx * cy * cz - sx * sz
        R[:, 0, 1] = -cx * cy * sz - sx * cz
        R[:, 0, 2] = cx * sy
        R[:, 1, 0] = sx * cy * cz + cx * sz
        R[:, 1, 1] = -sx * cy * sz + cx * cz
        R[:, 1, 2] = sx * sy
        R[:, 2, 0] = -sy * cz
        R[:, 2, 1] = sy * sz
        R[:, 2, 2] = cy

    elif convention == 'xyz':
        # R = Rz * Ry * Rx
        R = torch.zeros(angles.shape[0], 3, 3, device=angles.device, dtype=angles.dtype)
        R[:, 0, 0] = cy * cz
        R[:, 0, 1] = -cy * sz
        R[:, 0, 2] = sy
        R[:, 1, 0] = cx * sz + sx * sy * cz
        R[:, 1, 1] = cx * cz - sx * sy * sz
        R[:, 1, 2] = -sx * cy
        R[:, 2, 0] = sx * sz - cx * sy * cz
        R[:, 2, 1] = sx * cz + cx * sy * sz
        R[:, 2, 2] = cx * cy

    elif convention == 'zyx':
        # R = Rx * Ry * Rz
        R = torch.zeros(angles.shape[0], 3, 3, device=angles.device, dtype=angles.dtype)
        R[:, 0, 0] = cy * cz
        R[:, 0, 1] = sx * sy * cz - cx * sz
        R[:, 0, 2] = cx * sy * cz + sx * sz
        R[:, 1, 0] = cy * sz
        R[:, 1, 1] = sx * sy * sz + cx * cz
        R[:, 1, 2] = cx * sy * sz - sx * cz
        R[:, 2, 0] = -sy
        R[:, 2, 1] = sx * cy
        R[:, 2, 2] = cx * cy

    elif convention == 'xzy':
        # R = Ry * Rz * Rx
        R = torch.zeros(angles.shape[0], 3, 3, device=angles.device, dtype=angles.dtype)
        R[:, 0, 0] = cy * cz
        R[:, 0, 1] = -sz
        R[:, 0, 2] = sy * cz
        R[:, 1, 0] = sx * sy + cx * cy * sz
        R[:, 1, 1] = cx * cz
        R[:, 1, 2] = cx * sy * sz - sx * cy
        R[:, 2, 0] = sx * cy * sz - cx * sy
        R[:, 2, 1] = sx * cz
        R[:, 2, 2] = cx * cy + sx * sy * sz

    elif convention == 'yxz':
        # R = Rz * Rx * Ry
        R = torch.zeros(angles.shape[0], 3, 3, device=angles.device, dtype=angles.dtype)
        R[:, 0, 0] = cy * cz + sx * sy * sz
        R[:, 0, 1] = sx * sy * cz - cy * sz
        R[:, 0, 2] = cx * sy
        R[:, 1, 0] = cx * sz
        R[:, 1, 1] = cx * cz
        R[:, 1, 2] = -sx
        R[:, 2, 0] = sx * cy * sz - sy * cz
        R[:, 2, 1] = sy * sz + sx * cy * cz
        R[:, 2, 2] = cx * cy

    elif convention == 'yzx':
        # R = Rx * Rz * Ry
        R = torch.zeros(angles.shape[0], 3, 3, device=angles.device, dtype=angles.dtype)
        R[:, 0, 0] = cy * cz
        R[:, 0, 1] = sx * sy - cx * cy * sz
        R[:, 0, 2] = cx * sy + sx * cy * sz
        R[:, 1, 0] = sz
        R[:, 1, 1] = cx * cz
        R[:, 1, 2] = -sx * cz
        R[:, 2, 0] = -sy * cz
        R[:, 2, 1] = sx * cy + cx * sy * sz
        R[:, 2, 2] = cx * cy - sx * sy * sz

    elif convention == 'zxy':
        # R = Ry * Rx * Rz
        R = torch.zeros(angles.shape[0], 3, 3, device=angles.device, dtype=angles.dtype)
        R[:, 0, 0] = cy * cz - sx * sy * sz
        R[:, 0, 1] = -cx * sz
        R[:, 0, 2] = sy * cz + sx * cy * sz
        R[:, 1, 0] = cy * sz + sx * sy * cz
        R[:, 1, 1] = cx * cz
        R[:, 1, 2] = sy * sz - sx * cy * cz
        R[:, 2, 0] = -cx * sy
        R[:, 2, 1] = sx
        R[:, 2, 2] = cx * cy

    else:
        raise ValueError(f"Unsupported convention: {convention}")

    R = R.view(*batch_shape, 3, 3)
    return R