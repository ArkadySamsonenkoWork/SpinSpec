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


class BoundaryHandler:
    """Handles boundary condition logic."""
    @staticmethod
    def get_boundary(boundary: str | None, init_indexes: list[int], is_start: bool):
        if boundary == "reflection":
            offsets = (2, 1) if is_start else (-3, -2)
        elif boundary == "a":
            offsets = (-3, -2) if is_start else (1, 2)
        elif boundary is None:
            return []
        else:
            raise ValueError("Invalid phi boundary condition.")
        return [init_indexes[offset] for offset in offsets]


class ThetaLine:
    def __init__(self, theta: float, points: int, phi_limits: tuple[float, float],
                 last_point: bool):
        self.phi_limits = phi_limits
        self.theta = theta
        self.latent_points = points
        self.last_point = last_point

    def _compute_visible_points(self):
        if self.last_point:
            return self.latent_points
        return self.latent_points if self.latent_points == 1 else self.latent_points - 1

    def phi_theta(self):
        if self.latent_points == 1:
            return [(0.0, 0.0)]
        delta_phi = self.phi_limits[1] - self.phi_limits[0]
        if self.last_point:
            return [(self.phi_limits[0] + point * delta_phi / (self.latent_points - 1), self.theta) for point in
                    range(self.latent_points)]
        else:
            return [(self.phi_limits[0] + point * delta_phi / (self.latent_points - 1), self.theta) for point in
                    range(self.latent_points - 1)]



class RBFInterpolator:
    def __init__(self,
                 interpolating_indexes: list[int],
                 base_vertices: list[tuple[float, float]] | np.ndarray,
                 extended_vertices: list[tuple[float, float]],
                 kernel: str = "gaussian",
                 epsilon: float = 1.0):
        """
        Radial Basis Function (RBF) interpolator on the sphere.

        :param interpolating_indexes: Mapping of base mesh indices to actual data indices.
        :param base_vertices: List of (lat, lon) base mesh vertices in radians.
        :param extended_vertices: List of (lat, lon) extended mesh vertices in radians.
        :param kernel: Choice of kernel: "gaussian", "multiquadric", "inverse_multiquadric", "linear", "cubic", "thin_plate".
        :param epsilon: Shape parameter for kernels like Gaussian/MQ/IMQ.
        """
        self.kernel = kernel
        self.epsilon = epsilon

        base = self._to_lat_long(base_vertices)
        extended = self._to_lat_long(extended_vertices)

        self.base = base
        self.extended = extended
        self.interp_indexes = torch.as_tensor(interpolating_indexes)
        self.extended_size = extended.shape[0]

        dists = pairwise_distances(base, base, metric="haversine")
        self.K = self._rbf(dists)

        dists_ext = pairwise_distances(extended, base, metric="haversine")
        self.K_ext = self._rbf(dists_ext)

        self.K_torch = torch.tensor(self.K, dtype=torch.float32)
        self.K_ext_torch = torch.tensor(self.K_ext, dtype=torch.float32)

        jitter = 1e-4
        K = 0.5 * (self.K_torch.double() + self.K_torch.double().T)


        K = K + jitter * torch.eye(K.shape[0], dtype=K.dtype)
        U, S, Vh = torch.linalg.svd(K)

        S_inv = torch.where(S > 1e-10, 1.0 / S, torch.zeros_like(S))
        self.K_inv = (Vh.mT @ torch.diag(S_inv) @ U.mT).to(torch.float32)



    def _to_lat_long(self, array: list[tuple[float, float]]):
        array = np.array(array)[:, ::-1]
        array[:, 0] = np.pi / 2 - array[:, 0]
        return array

    def _rbf(self, r: np.ndarray) -> np.ndarray:
        """Radial basis functions."""
        eps = self.epsilon
        if self.kernel == "gaussian":
            return np.exp(-(eps * r) ** 2)
        elif self.kernel == "multiquadric":
            return np.sqrt(1.0 + (eps * r) ** 2)
        elif self.kernel == "inverse_multiquadric":
            return 1.0 / np.sqrt(1.0 + (eps * r) ** 2)
        elif self.kernel == "linear":
            return r
        elif self.kernel == "cubic":
            return r ** 3
        elif self.kernel == "thin_plate":
            return (r ** 2) * np.log(r + 1e-6)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def __call__(self, f_values: torch.Tensor) -> torch.Tensor:
        """
        Interpolate values at extended points using RBF interpolation.

        :param f_values: Tensor of shape (..., N), where N = number of base vertices.
        :return: Interpolated values of shape (..., M), where M = number of extended vertices.
        """

        y = f_values.mT

        alpha = self.K_inv @ y
        f_extended = (self.K_ext_torch @ alpha).mT

        return f_extended


class Kernels:
    @staticmethod
    def rbf(kernel: str, r: np.ndarray, epsilon: float) -> np.ndarray:
        """Radial basis functions."""
        eps = epsilon
        if kernel == "gaussian":
            return np.exp(-(eps * r) ** 2)
        elif kernel == "inverse_multiquadric":
            return 1.0 / np.sqrt(1.0 + (eps * r) ** 2)
        elif kernel == "linear":
            return 1 / (r + 1e-8 * eps)
        elif kernel == "quadratic":
            return 1 / (np.log(r + 1e-9 * eps) * (r + 1e-8 * eps) ** 2)
        elif kernel == "inverse_thin_plate":
            return 1 / ((r + 1e-9 * eps) * np.log(r + 1e-9))
        else:
            raise ValueError(f"Unknown kernel: {kernel}")


class NearestNeighborsInterpolator(nn.Module):
    def __init__(self,
                 init_vertices: list[tuple[float, float]],
                 extended_vertices: list[tuple[float, float]],
                 k: int = 4, device: torch.device = torch.device("cpu")):
        """

        Initialize the interpolator with the base mesh vertices and extended mesh vertices.
        Uses a BallTree for efficient nearest neighbor search.
        """
        super().__init__()
        self.k = k

        tree = BallTree(self._to_lat_long(init_vertices), metric="haversine")
        distances, indexes = tree.query(self._to_lat_long(extended_vertices), k=self.k)

        clipped = np.clip(distances, a_min=1e-9, a_max=None)


        inv_distances = torch.tensor(Kernels().rbf(kernel="linear",
                                                   r=clipped, epsilon=1), dtype=torch.float32, device=device)
        weights = inv_distances / inv_distances.sum(dim=-1, keepdim=True)
        self.register_buffer("weights", weights)

        indexes = torch.as_tensor(indexes, device=device)
        self.register_buffer("indexes", indexes)

    def _to_lat_long(self, array: list[tuple[float, float]]):
        array = np.array(array)[:, ::-1]  # theta, phi
        array[:, 0] = np.pi/2 - array[:, 0]
        return array

    def forward(self, f_values: torch.Tensor) -> torch.Tensor:
        """
        Interpolate values at extended points using inverse distance weighting.
        :param f_values: Tensor of shape (..., N), where N is the number of base vertices.
        :return: Interpolated values of shape (..., M), where M is the number of extended vertices.
        """

        """
        shape = f_values.shape
        f_extended = torch.zeros((*shape[:-1], self.extended_size), dtype=f_values.dtype)
        mapped_indexes = self.interp_indexes[self.indexes]

        for idx in range(self.k):
            f_extended += f_values[..., mapped_indexes[..., idx]] * self.weights[..., idx]
        return f_extended
        """

        orig_shape = f_values.shape
        batch_shape = orig_shape[:-1]
        orig_size = orig_shape[-1]

        mapped = self.indexes.long().to(f_values.device)
        weights = self.weights.to(f_values.device)

        if mapped.ndim == 2:
            mapped = mapped.unsqueeze(0).expand(*batch_shape, mapped.shape[-2], mapped.shape[-1])
        elif mapped.ndim == f_values.ndim - 1:
            mapped = mapped.unsqueeze(0).expand(*batch_shape, mapped.shape[-2], mapped.shape[-1])
        else:
            if mapped.shape[:-2] != batch_shape:
                mapped = mapped.expand(*batch_shape, mapped.shape[-2], mapped.shape[-1])
        ext_size = mapped.shape[-2]
        k = mapped.shape[-1]
        expanded = f_values.unsqueeze(-2).expand(*batch_shape, ext_size, orig_size)
        gathered = torch.take_along_dim(expanded, mapped, dim=-1)
        return (gathered * weights).sum(dim=-1).transpose(-1, -2)


class MeshProcessorBase(nn.Module):
    def __init__(self, init_grid_frequency, phi_limits, boundaries_cond,
                device: torch.device = torch.device("cpu")):
        super().__init__()
        self.init_grid_frequency = init_grid_frequency
        self.phi_limits = phi_limits
        self.boundaries_cond = boundaries_cond
        self.last_point = boundaries_cond != "periodic"

    def _create_theta_lines(self, grid_frequency: int, last_point: bool):
        eps = 1e-8
        init_lines = [ThetaLine(
                theta=0.0,
                points=1,
                phi_limits=self.phi_limits,
                last_point=last_point,
        ),
            ThetaLine(
                theta=eps,
                points=2,
                phi_limits=self.phi_limits,
                last_point=last_point,
        ),
        ThetaLine(
                theta=2 * eps,
                points=3,
                phi_limits=self.phi_limits,
                last_point=last_point,
        )
        ]

        return init_lines + [
            ThetaLine(
                theta=np.arccos(1 - (point - 3)**2 / (grid_frequency - 3)**2),
                points=point,
                phi_limits=self.phi_limits,
                last_point=last_point,
            ) for point in range(4, grid_frequency + 1)
        ]


    def _assemble_vertices(self, theta_lines):
        return np.concatenate(
            [np.array(tl.phi_theta(), dtype=np.float32) for tl in theta_lines],
            axis=0
        )

    def _create_triangular_dict(self, K: int):
        """Vectorized version using NumPy operations"""
        i_vals = np.arange(K)
        i_grid, j_grid = np.meshgrid(i_vals, i_vals, indexing='ij')

        mask = j_grid <= i_grid
        i_tri = i_grid[mask]
        j_tri = j_grid[mask]
        return {(i, j): idx for idx, (i, j) in enumerate(zip(i_tri, j_tri))}

    def _build_triangles(self, K: int):
        positions = []
        indices = []
        for i in range(K):
            for j in range(i + 1):
                positions.append((i, j))
                indices.append(i * (i + 1) // 2 + j)

        pos_to_idx = {pos: idx for pos, idx in zip(positions, indices)}
        upward_specs = []
        downward_specs = []
        for k in range(K - 1):
            for q in range(k + 1):
                upward_specs.append([(k, q), (k + 1, q), (k + 1, q + 1)])

        for k in range(1, K - 1):
            for q in range(1, k + 1):
                downward_specs.append([(k, q), (k, q - 1), (k + 1, q)])

        all_specs = upward_specs + downward_specs
        triangles = np.array([[pos_to_idx[spec[0]], pos_to_idx[spec[1]], pos_to_idx[spec[2]]]
                              for spec in all_specs], dtype=int)

        return triangles

    def _triangulate(self, grid_frequency: int):
        triangulation = self._build_triangles(grid_frequency)
        return triangulation


class InterpolatingMeshProcessor(MeshProcessorBase):
    def __init__(self, interpolate_grid_frequency, device: torch.device = torch.device("cpu"), *args, **kwargs):
        super().__init__(device=device, *args, **kwargs)
        self.interpolate_grid_frequency = interpolate_grid_frequency

        self.base_theta_lines = self._create_theta_lines(self.init_grid_frequency, last_point=True)
        self.interpolating_theta_lines = self._create_theta_lines(self.interpolate_grid_frequency, last_point=True)

        self.final_vertices, self.simplices = self._get_post_mesh()

        self.init_vertices = self._assemble_vertices(self.base_theta_lines)
        self.interpolator = self._get_interpolator(self.final_vertices, device=device)

        self.extended_size = self.final_vertices.shape[0]

    def _get_post_mesh(self):
        extended_vertices = self._assemble_vertices(self.interpolating_theta_lines)
        simplices = self._triangulate(self.interpolate_grid_frequency)
        return extended_vertices, simplices

    def _get_interpolator(self, extended_vertices, device: torch.device):
        return NearestNeighborsInterpolator(init_vertices=self.init_vertices,
                                            extended_vertices=self.final_vertices, device=device)

    def forward(self, f_values: torch.Tensor):
        shape = f_values.shape
        init_vert_dim = shape[-1]
        f_new = f_values.reshape((-1, init_vert_dim))
        out = self.interpolator(f_new).transpose(-1, -2)
        return out.reshape((*shape[:-1], out.shape[-1]))


class BoundaryMeshProcessor(MeshProcessorBase):
    def __init__(self, device: torch.device = torch.device("cpu"), *args, **kwargs):
        super().__init__(device=device, *args, **kwargs)

        self.base_theta_lines = self._create_theta_lines(self.init_grid_frequency, last_point=True)
        self.final_vertices, self.simplices = self._get_post_mesh()

        self.init_vertices = self._assemble_vertices(self.base_theta_lines)
        self.extended_size = self.final_vertices.shape[0]

    def _get_post_mesh(self):
        vertices = self._assemble_vertices(self.base_theta_lines)
        simplices = self._triangulate(self.init_grid_frequency)
        return vertices, simplices

    def forward(self, f_values: torch.Tensor):
        return f_values


def mesh_processor_factory(init_grid_frequency,
                           interpolate_grid_frequency,
                           interpolate=False,
                           boundaries_cond=None,
                           phi_limits=(0, 2 * math.pi),
                           device: torch.device = torch.device("cpu")):

    if interpolate:
        return InterpolatingMeshProcessor(
            interpolate_grid_frequency=interpolate_grid_frequency,
            init_grid_frequency=init_grid_frequency,
            phi_limits=phi_limits,
            boundaries_cond=boundaries_cond, device=device
        )
    elif boundaries_cond != "periodic":
        return BoundaryMeshProcessor(
            init_grid_frequency=init_grid_frequency,
            phi_limits=phi_limits,
            boundaries_cond=boundaries_cond, device=device
        )


class DelaunayMeshNeighbour(BaseMeshPowder):
    """Delaunay triangulation-based spherical mesh implementation."""
    """It uses Close Neighbour method to interpolate"""
    def __init__(self,
                 eps: float = 1e-7,
                 phi_limits: tuple[float, float] = (0, 2 * math.pi),
                 initial_grid_frequency: int = 20,
                 interpolation_grid_frequency: int = 40,
                 boundaries_cond=None,
                 interpolate=False,
                 dtype=torch.float32, device: torch.device = torch.device("cpu")):
        """
        Initialize Delaunay mesh parameters.

        Args:
            eps: Small epsilon value for numerical stability
            phi_limits: Maximum value for phi coordinate (default: full circle)
            initial_grid_frequency: Resolution of initial grid
            interpolation_grid_frequency: Resolution of interpolation grid
        """
        super().__init__(device=device, dtype=dtype)
        self.dtype = dtype
        self.eps = eps

        self.phi_limit = phi_limits
        self.initial_grid_frequency = initial_grid_frequency
        if interpolate:
            self.interpolation_grid_frequency = interpolation_grid_frequency
        else:
            self.interpolation_grid_frequency = initial_grid_frequency
        self.mesh_processor = mesh_processor_factory(initial_grid_frequency, interpolation_grid_frequency,
                                                     device=device,
                                                     phi_limits=phi_limits, interpolate=interpolate,
                                                     boundaries_cond=boundaries_cond)

        (initial_grid,
         post_grid,
         post_simplices) = self.create_initial_cache_data(device)

        self.register_buffer("_initial_grid", initial_grid)
        self.register_buffer("_post_grid", post_grid)
        self.register_buffer("_post_simplices", post_simplices)
        self.to(device)

    def create_initial_cache_data(self, device: torch.device) -> tuple:
        """Create and cache initial mesh data structures."""
        return (
            torch.as_tensor(self.mesh_processor.init_vertices, dtype=self.dtype, device=device),
            torch.as_tensor(self.mesh_processor.final_vertices, dtype=self.dtype, device=device),
            torch.as_tensor(self.mesh_processor.simplices, device=device)
        )

    def _triangulate(self, vertices: np.ndarray) -> Delaunay:
        """Perform Delaunay triangulation on given vertices."""
        return Delaunay(vertices)

    @property
    def post_mesh(self):
        return self._post_grid, self._post_simplices

    @property
    def initial_grid(self):
        return self._initial_grid

    def to_delaunay(self,
                    f_post: torch.Tensor,
                    simplices: torch.Tensor) -> torch.Tensor:
        """
        Format interpolated values for Delaunay representation.

        Args:
            f_post: Interpolated function values
            simplices: Simplices to use for final representation

        Returns:
            torch.Tensor: Values formatted for Delaunay triangulation
        """
        return f_post[..., simplices]

    def forward(self,
                    f_init: torch.Tensor) -> torch.Tensor:
        """
        Format interpolated values for Delaunay representation.

        Args:
            f_init: Interpolated function values

        Returns:
            torch.Tensor: Values formatted for Delaunay triangulation
        """
        return self.mesh_processor(f_init)


class DelaunayMeshNeighbourFullSphere(DelaunayMeshNeighbour):
    def __init__(self,
                 eps: float = 1e-7,
                 phi_limits: tuple[float, float] = (0, 2 * math.pi),
                 initial_grid_frequency: int = 20,
                 interpolation_grid_frequency: int = 40,
                 boundaries_cond=None,
                 interpolate=False,
                 dtype=torch.float32, device: torch.device = torch.device("cpu")):
        super().__init__(eps, phi_limits, initial_grid_frequency,
                         interpolation_grid_frequency, boundaries_cond, interpolate, dtype, device
                         )
        _second_unit = self._initial_grid.clone()

        _second_unit[:, 1] = torch.pi - _second_unit[:, 1]
        self._initial_grid = torch.cat((self._initial_grid, _second_unit), dim=-2)

        _second_unit = self._post_grid.clone()
        _second_unit[:, 1] = torch.pi - _second_unit[:, 1]
        self._post_grid = torch.cat((self._post_grid, _second_unit), dim=-2)

        second_simpl = self._post_simplices.clone()
        second_simpl = second_simpl + _second_unit.shape[-2]
        self._post_simplices = torch.cat((self._post_simplices, second_simpl), dim=-2)