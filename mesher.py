from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.spatial import Delaunay

class Mesh(ABC):
    @abstractmethod
    def __init__(self, eps: float = 1e-5, phi_limit: float = 2 * np.pi,
                 initial_grid_frequency: int = 10, interpolation_grid_frequency: int = 50):
        pass

    def create_rotation_matrices(self, grid):
        """
        Given tensors phi and theta (of the same shape), returns a tensor
        of shape (..., 3, 3) where each 3x3 matrix rotates the z-axis to the direction
        defined by the spherical angles (phi, theta).

        The rotation is computed as R = R_z(phi) @ R_y(theta), where:
          R_z(phi) = [[cos(phi), -sin(phi), 0],
                      [sin(phi),  cos(phi), 0],
                      [      0,         0, 1]]
          R_y(theta) = [[cos(theta), 0, sin(theta)],
                        [         0, 1,          0],
                        [-sin(theta), 0, cos(theta)]]
        """
        phi = grid[..., 0]
        theta = grid[..., 1]
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        R = torch.empty(*phi.shape, 3, 3, dtype=phi.dtype, device=phi.device)

        R[..., 0, 0] = cos_phi * cos_theta
        R[..., 0, 1] = -sin_phi
        R[..., 0, 2] = cos_phi * sin_theta

        R[..., 1, 0] = sin_phi * cos_theta
        R[..., 1, 1] = cos_phi
        R[..., 1, 2] = sin_phi * sin_theta

        # Third row
        R[..., 2, 0] = -sin_theta
        R[..., 2, 1] = 0.0
        R[..., 2, 2] = cos_theta

        return R

    @abstractmethod
    def interpolate(self, f_values: torch.Tensor):
        pass

    @abstractmethod
    def to_delaunay(self, f_interpolated: torch.Tensor):
        pass


class DelaunayMesh(Mesh):
    def __init__(self, eps: float = 1e-5, phi_limit: float = 2 * np.pi,
                 initial_grid_frequency: int = 10, interpolation_grid_frequency: int = 50):
        """
        :param eps:
        """
        self.eps = np.array(eps)
        self.phi_limit = phi_limit
        self.initial_grid_frequency = initial_grid_frequency
        self.interpolation_grid_frequency = interpolation_grid_frequency
        self.initial_grid, self.interpolation_grid, self.interpolation_indexes,\
            self.bary_coords, self.final_simplices = \
            self.create_initial_cash_data()

    def create_initial_cash_data(self):
        initial_grid = self.create_grid(self.initial_grid_frequency, self.phi_limit)
        initial_tri = self._triangulate(initial_grid)
        interpolation_grid = self.create_grid(self.interpolation_grid_frequency, self.phi_limit)
        interpolation_tri = self._triangulate(interpolation_grid)
        final_simplices = interpolation_tri.simplices
        interpolation_indexes, bary_coords = self._get_interpolation_coeffs(interpolation_grid, initial_tri)

        initial_grid = torch.as_tensor(initial_grid)
        interpolation_indexes = torch.as_tensor(interpolation_indexes)
        bary_coords = torch.as_tensor(bary_coords)
        final_simplices = torch.as_tensor(final_simplices)
        return initial_grid, interpolation_grid, interpolation_indexes, bary_coords, final_simplices


    def create_grid(self, grid_frequency: int = 10, phi_limit: float = 2 * np.pi):
        """
        Create grid points using the parameterization:
        K == grid_frequency
           theta = (pi/2)*(k/K - 1)   for k = 0,1,...,K      (θ in [-pi/2, 0])
           phi   = (pi/2)*(q/k)       for q = 0,1,...,k
        For k == 0, we only have one point (q==0).
        """
        self.phi_limit = phi_limit
        vertices = [(0.0, 0.0), (phi_limit, 0)] +\
                   [(phi_limit * (q / k), (np.pi / 2) * (k / (grid_frequency - 1)))
                    for k in range(grid_frequency) for q in range(k+1)]
        return vertices

    def _triangulate(self, vertices):
        """
        Given an array of vertices (phi, theta), build a Delaunay triangulation.
        """
        tri = Delaunay(vertices)
        return tri

    def _get_simplex_indexes(self, qp_np: np.array, tri: Delaunay):
        """
        :param qp_np: array of new phi, theta points
        :param tri: Delaunay triangulation
        :return: indexes of Delaunay triangulation of the original mesh
        """
        qp_np_search = qp_np.copy()
        mask = qp_np_search[:, 0] == 0
        qp_np_search[mask, 0] += self.eps
        mask = qp_np_search[:, 0] == np.pi / 2
        qp_np_search[mask] -= self.eps
        mask = qp_np_search[:, 1] == 0
        qp_np_search[mask] += self.eps
        mask = qp_np_search[:, 1] == np.pi / 2
        qp_np_search[mask] -= self.eps

        simplex_indices = tri.find_simplex(qp_np_search)
        return simplex_indices

    def reflect(self, x, interval_begin, interval_end):
        """
        Reflect each element of tensor x into the interval [interval_begin, interval_end].
        (For example, if x > interval_end then x is replaced by 2*interval_end - x.)
        This works even if x is several periods outside the interval.
        """
        L = interval_end - interval_begin
        x_shifted = x - interval_begin
        mod = 2 * L
        x_mod = torch.remainder(x_shifted, mod)
        x_reflected = torch.where(x_mod > L, mod - x_mod, x_mod)
        return interval_begin + x_reflected

    def _get_interpolation_coeffs(self, query_points, tri):
        """
        Given:
          - query_points: (N,2) array of (phi, theta) at which to evaluate f,
          - vertices: (M,2) array of grid points (phi, theta),
          - f_values: (M,) array of function values at those vertices,
          - tri: Delaunay triangulation of vertices,
        perform:
          - Reflection of query_points into the grid domain,
          - For each query point, find the containing simplex and compute barycentric coordinates,
          - For query points outside the triangulation (simplex index == -1), perform nearest–neighbor interpolation.
        Returns:
          A torch.Tensor of shape (N,) with the interpolated values.
        """
        phi_min, phi_max = 0.0, self.phi_limit
        theta_min, theta_max = 0.0, np.pi / 2

        qp = torch.as_tensor(query_points, dtype=torch.float32)

        qp[:, 0] = self.reflect(qp[:, 0], phi_min, phi_max)
        qp[:, 1] = self.reflect(qp[:, 1], theta_min, theta_max)

        qp_np = qp.detach().cpu().numpy()
        simplex_indices = self._get_simplex_indexes(qp_np, tri)


        simplices = tri.simplices[simplex_indices]
        T = tri.transform[simplex_indices, :2, :]
        r = qp_np - tri.transform[simplex_indices, 2, :]
        bary = np.einsum('nij,nj->ni', T, r)  # (n_valid,2)
        bary_coords = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))  # shape (n, 3)

        return simplices, bary_coords

    def interpolate(self, f_values: torch.Tensor):
        """
        :param f_values: the function to interpolate. The shape is [..., M]
        where M is the mesh at the initial points
        :return: f_interpolated
        """
        f_verts = f_values[self.interpolation_indexes]
        return (f_verts * self.bary_coords).sum(dim=1)

    def to_delaunay(self, f_interpolated: torch.Tensor):
        """
        :param f_interpolated: the function to interpolate. The shape is [..., N]
        where M is the mesh at the initial points
        :return: f_delaunay: The shape is [..., K, 3]. Transform data into Delaunay triangulation
        """
        return f_interpolated[self.final_simplices]

    def _angle_between(self, u, v):
        dot = (u * v).sum(dim=1)
        # Clamp the dot product to avoid numerical errors outside [-1, 1]
        dot = torch.clamp(dot, -1.0, 1.0)
        return torch.acos(dot)

    def spherical_triangle_areas(self, vertices, triangles):
        """
        vertices: tensor of shape (N,2), each row is [phi, theta]
        triangles: tensor of shape (M,3) with indices into vertices defining the triangles.

        Returns:
           areas: tensor of shape (M,) with the spherical areas of the triangles (for unit sphere).
                  For a sphere of radius R, multiply each area by R**2.
        """
        phi = vertices[:, 0]
        theta = vertices[:, 1]
        x = torch.cos(theta) * torch.cos(phi)
        y = torch.cos(theta) * torch.sin(phi)
        z = torch.sin(theta)
        xyz = torch.stack([x, y, z], dim=1)  # shape (N, 3)

        v0 = xyz[triangles[:, 0]]
        v1 = xyz[triangles[:, 1]]
        v2 = xyz[triangles[:, 2]]

        a = self._angle_between(v1, v2)
        b = self._angle_between(v2, v0)
        c = self._angle_between(v0, v1)

        alpha = torch.acos(
            torch.clamp((torch.cos(a) - torch.cos(b) * torch.cos(c)) / (torch.sin(b) * torch.sin(c)), -1.0, 1.0))
        beta = torch.acos(
            torch.clamp((torch.cos(b) - torch.cos(c) * torch.cos(a)) / (torch.sin(c) * torch.sin(a)), -1.0, 1.0))
        gamma = torch.acos(
            torch.clamp((torch.cos(c) - torch.cos(a) * torch.cos(b)) / (torch.sin(a) * torch.sin(b)), -1.0, 1.0))

        excess = (alpha + beta + gamma) - torch.pi

        return excess


