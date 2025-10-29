from abc import ABC, abstractmethod
import sys
import typing as tp
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

sys.path.append("..")
import utils


class BaseMesh(nn.Module, ABC):
    @abstractmethod
    def __init__(self, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__()
        self.register_buffer("_rotation_matrices", None)

    @property
    @abstractmethod
    def rotation_matrices(self):
        pass

    @property
    @abstractmethod
    def initial_size(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class CrystalMesh(BaseMesh):
    def __init__(self, euler_angles: torch.Tensor,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32,
                 convention: str = "zyz"):
        """
        :param euler_angles: torch.Tensor of shape (..., 3) containing Euler angles in radians
        :param convention: str, rotation convention (default 'xyz')
                       Supported: 'zyz', 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
        :return: torch.Tensor of shape (..., 3, 3) containing rotation matrices
        """
        super().__init__(device=device)
        if euler_angles.dim() == 1:
            euler_angles = euler_angles.unsqueeze(0)
        self.register_buffer("_rotation_matrices",
                             utils.euler_angles_to_matrix(euler_angles.to(device=device, dtype=dtype), convention)
                             )

    @property
    def rotation_matrices(self):
        return self._rotation_matrices

    @property
    def initial_size(self):
        return self.rotation_matrices.shape[:-2]

    @property
    def name(self) -> str:
        return "CrystalMesh"


class BaseMeshPowder(BaseMesh):
    @abstractmethod
    def __init__(self, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(device=device, dtype=dtype)
        self._rotation_matrices: tp.Optional[torch.Tensor] = None

    @property
    def rotation_matrices(self):
        if self._rotation_matrices is None:
            self._rotation_matrices = self._create_rotation_matrices()
        return self._rotation_matrices

    @property
    def initial_size(self):
        return self.initial_grid.size()[:-1]

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
        phi = self.initial_grid[..., 0]
        theta = self.initial_grid[..., 1]
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        R = torch.empty(*phi.shape, 3, 3, dtype=phi.dtype, device=phi.device)

        R[..., 0, 0] = cos_phi * cos_theta
        R[..., 0, 1] = -sin_phi * cos_theta
        R[..., 0, 2] = sin_theta

        R[..., 1, 0] = sin_phi
        R[..., 1, 1] = cos_phi
        R[..., 1, 2] = 0

        # Third row
        R[..., 2, 0] = -sin_theta * cos_phi
        R[..., 2, 1] = sin_theta * sin_phi
        R[..., 2, 2] = cos_theta
        return R

    def areas(self):
        vertices, triangles = self.post_mesh
        return self.spherical_triangle_areas(vertices, triangles)

    @staticmethod
    def spherical_triangle_areas(vertices: torch.Tensor, triangles: torch.Tensor):
        """
        vertices: tensor of shape (N,2), each row is [phi, theta]
        triangles: tensor of shape (M,3) with indices into vertices defining the triangles.

        Returns:
           areas: tensor of shape (M,) with the spherical areas of the triangles (for unit sphere).
                  For a sphere of radius R, multiply each area by R**2.
        """
        def _angle_between(u, v):
            dot = (u * v).sum(dim=1)
            dot = torch.clamp(dot, -1.0, 1.0)
            return torch.acos(dot)
        phi = vertices[:, 0]
        theta = vertices[:, 1]

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        xyz = torch.stack([x, y, z], dim=1)

        v0 = xyz[triangles[:, 0]]
        v1 = xyz[triangles[:, 1]]
        v2 = xyz[triangles[:, 2]]

        a = _angle_between(v1, v2)
        b = _angle_between(v2, v0)
        c = _angle_between(v0, v1)

        s = (a + b + c) / 2

        # L'Huilier's formula for spherical excess
        tan_E_4 = torch.sqrt(
            torch.clamp(
                torch.tan(s / 2) * torch.tan((s - a) / 2) * torch.tan((s - b) / 2) * torch.tan((s - c) / 2),
                min=0.0
            )
        )

        excess = 4 * torch.atan(tan_E_4)
        return excess

    @property
    @abstractmethod
    def initial_grid(self):
        pass

    @property
    @abstractmethod
    def post_mesh(self):
        pass

    @abstractmethod
    def to_delaunay(self, f_interpolated: torch.Tensor, simplices: torch.Tensor):
        pass

    @abstractmethod
    def forward(self, f_function: torch.Tensor):
        pass

    @property
    def name(self) -> str:
        return "PowderMesh"

    def triplot(self):
        mesh, triplots = self.post_mesh
        phi, theta = mesh[..., 0], mesh[..., 1]
        plt.triplot(phi.numpy(), theta.numpy(), triplots)



class BaseMeshAxial(BaseMeshPowder):
    @abstractmethod
    def __init__(self, device: torch.device = torch.device("cpu"), *args, **kwargs):
        super().__init__(device=device, *args, **kwargs)

    @property
    def initial_size(self):
        return self.initial_grid.size()[:-1]

    def _create_rotation_matrices(self):
        """
        Given tensors phi and theta (of the same shape), returns a tensor
        of shape (..., 3, 3) where each 3x3 matrix rotates the z-axis to the direction
        defined by the spherical angles (theta).

        The rotation is computed as R =  R_y(theta), where:
          R_y(theta) = [[cos(theta), 0, sin(theta)],
                        [         0, 1,          0],
                        [-sin(theta), 0, cos(theta)]]
        """
        theta = self.initial_grid[..., 0]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        R = torch.empty(*theta.shape, 3, 3, dtype=theta.dtype, device=theta.device)

        R[..., 0, 0] = cos_theta
        R[..., 0, 1] = 0.0
        R[..., 0, 2] = sin_theta

        R[..., 1, 0] = 0.0
        R[..., 1, 1] = 1.0
        R[..., 1, 2] = 0

        # Third row
        R[..., 2, 0] = -sin_theta
        R[..., 2, 1] = 0.0
        R[..., 2, 2] = cos_theta
        return R

    @staticmethod
    def spherical_triangle_areas(vertices: torch.Tensor, triangles: torch.Tensor):
        """
        vertices: tensor of shape (N,1), each row is [theta]
        triangles: tensor of shape (M,2) with indices into vertices defining the lines.

        Returns:
           areas: tensor of shape (M,) with the spherical areas of the triangles (for unit sphere).
                  For a sphere of radius R, multiply each area by R**2.
        """
        theta = vertices[:, 0]
        end_theta = theta[triangles[:, 1]]
        start_theta = theta[triangles[:, 0]]
        excess = 2 * torch.pi * (torch.cos(end_theta) - torch.cos(start_theta))

        return excess

    @property
    @abstractmethod
    def initial_grid(self):
        pass

    @property
    @abstractmethod
    def post_mesh(self):
        pass

    @abstractmethod
    def to_delaunay(self, f_interpolated: torch.Tensor, simplices: torch.Tensor):
        pass

    @abstractmethod
    def forward(self, f_function: torch.Tensor):
        pass