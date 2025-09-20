import numpy as np
import math

from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.neighbors import BallTree
from sklearn.metrics import pairwise_distances
from .general_mesh import BaseMesh, BaseMeshPowder

from .torch_tps import ThinPlateSpline

from . import delanay_neigbour


class Mesh3D(delanay_neigbour.DelaunayMeshNeighbour):
    def __init__(self,
                 eps: float = 1e-7,
                 phi_limits: tuple[float, float] = (0, 2 * math.pi),
                 initial_grid_frequency: int = 20,
                 interpolation_grid_frequency: int = 40,
                 boundaries_cond=None,
                 interpolate=False,
                 dtype=torch.float32, device: torch.device = torch.device("cpu"),
                 gamma_size=20):
        """
        Initialize Delaunay mesh parameters with gamma angle.

        Args:
            eps: Small epsilon value for numerical stability
            phi_limits: Maximum value for phi coordinate (default: full circle)
            initial_grid_frequency: Resolution of initial grid
            interpolation_grid_frequency: Resolution of interpolation grid
        """
        super().__init__(eps=eps,
                         phi_limits=phi_limits,
                         initial_grid_frequency=initial_grid_frequency,
                         interpolation_grid_frequency=interpolation_grid_frequency,
                         boundaries_cond=boundaries_cond,
                         interpolate=interpolate,
                         dtype=dtype,
                         device=device)
        self.gamma_angles = torch.tensor([(2 * i * torch.pi) / (gamma_size-1) for i in range(gamma_size)],
                                         device=device)

    def _create_rotation_matrices(self):
        """
        Given tensors phi and theta (of the same shape), returns a tensor
        of shape (..., 3, 3) where each 3x3 matrix rotates the z-axis to the direction
        defined by the spherical angles (phi, theta).

        The rotation is computed as R =  R_y(theta) @ R_z(phi), where:
          R_z(phi) = [[cos(phi), -sin(phi), 0],
                      [sin(phi),  cos(phi), 0],
                      [      0,         0, 1]]
          R_y(theta) = [[cos(theta), 0, sin(theta)],
                        [         0, 1,          0],
                        [-sin(theta), 0, cos(theta)]]
        """
        phi = self.initial_grid[..., 0][None, :].expand(len(self.gamma_angles), -1)
        theta = self.initial_grid[..., 1][None, :].expand(len(self.gamma_angles), -1)
        gamma = self.gamma_angles.unsqueeze(-1).expand(-1, len(self.initial_grid))

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        R = torch.empty(*phi.shape, 3, 3, dtype=phi.dtype, device=phi.device)

        R[..., 0, 0] = cos_gamma * cos_theta * cos_phi - sin_gamma * sin_phi
        R[..., 0, 1] = -cos_gamma * cos_theta * sin_phi - sin_gamma * cos_phi
        R[..., 0, 2] = cos_gamma * sin_theta

        R[..., 1, 0] = sin_gamma * cos_theta * cos_phi + cos_gamma * sin_phi
        R[..., 1, 1] = -sin_gamma * cos_theta * sin_phi + cos_gamma * cos_phi
        R[..., 1, 2] = sin_gamma * sin_theta

        R[..., 2, 0] = -sin_theta * cos_phi
        R[..., 2, 1] = sin_theta * sin_phi
        R[..., 2, 2] = cos_theta

        return R