import numpy as np
import math

from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.neighbors import BallTree

from .general_mesh import BaseMesh


class BoundaryHandler:
    """Handles boundary condition logic."""
    @staticmethod
    def get_boundary(boundary: str | None, init_indexes: list[int], is_start: bool):
        if boundary == "reflection":
            offsets = (2, 1) if is_start else (-3, -2)
        elif boundary == "periodic":
            offsets = (-3, -2) if is_start else (1, 2)
        elif boundary is None:
            return []
        else:
            raise ValueError("Invalid phi boundary condition.")
        return [init_indexes[offset] for offset in offsets]


class ThetaLine:
    def __init__(self, theta: float, points: int, phi_limits: tuple[float, float],
                 last_point: bool, boundaries_cond: str | None):
        self.phi_limits = phi_limits
        self.theta = theta
        self.latent_points = points
        self.last_point = last_point
        self.boundaries_cond = boundaries_cond
        self.visible_points = self._compute_visible_points()

    def _compute_visible_points(self):
        if self.last_point:
            return self.latent_points
        return self.latent_points if self.latent_points == 1 else self.latent_points - 1

    def _circular_index(self, lst, idx):
        return lst[idx % len(lst)]

    def get_boundary(self, boundary, init_indexes, is_start):
        if boundary == "reflection":
            return [
                self._circular_index(init_indexes, 2 if is_start else -3),
                self._circular_index(init_indexes, 1 if is_start else -2),
            ]
        elif boundary == "periodic":
            return [
                self._circular_index(init_indexes, -3 if is_start else 1),
                self._circular_index(init_indexes, -2 if is_start else 2),
            ]
        elif boundary is None:
            return []
        else:
            raise ValueError("Invalid phi boundary condition. Must be 'reflection', 'periodic', or None.")

    def interpolating_indexes(self):
        if self.last_point:
            init_indexes = list(range(self.latent_points))
        else:
            init_indexes = list(range(self.latent_points - 1))
            init_indexes.append(0)

        prefix = self.get_boundary(self.boundaries_cond, init_indexes, True)
        suffix = self.get_boundary(self.boundaries_cond, init_indexes, False)
        return prefix + init_indexes + suffix

    def interpolating_phi_theta(self):
        if self.latent_points == 1:
            return 5 * [(0.0, 0.0)] if self.boundaries_cond != None else [(0.0, 0.0)]
        delta_phi = self.phi_limits[1] - self.phi_limits[0]
        if self.boundaries_cond is not None:
            prefix = [
                (self.phi_limits[0] + point * delta_phi / (self.latent_points - 1), self.theta) for point in [-2, -1]
            ]
            suffix = [
                (self.phi_limits[0] + point * delta_phi / (self.latent_points - 1), self.theta) for point in
                [self.latent_points, self.latent_points + 1]
            ]
        else:
            prefix = []
            suffix = []

        return prefix + \
            [(self.phi_limits[0] + point * delta_phi / (self.latent_points - 1), self.theta) for point in
             range(self.latent_points)] + suffix

    def init_phi_theta(self):
        if self.latent_points == 1:
            return [(0.0, 0.0)]
        delta_phi = self.phi_limits[1] - self.phi_limits[0]
        if self.last_point:
            return [(self.phi_limits[0] + point * delta_phi / (self.latent_points - 1), self.theta) for point in
                    range(self.latent_points)]
        else:
            return [(self.phi_limits[0] + point * delta_phi / (self.latent_points - 1), self.theta) for point in
                    range(self.latent_points - 1)]

    def post_phi_theta(self):
        if self.latent_points == 1:
            return [(0.0, 0.0)]
        delta_phi = self.phi_limits[1] - self.phi_limits[0]
        return [(self.phi_limits[0] + point * delta_phi / (self.latent_points - 1), self.theta) for point in
                range(self.latent_points)]

    def post_indexes(self):
        if self.last_point:
            return list(range(self.latent_points))
        else:
            return list(range(self.latent_points - 1)) + [0]

    def triangulated_vertices(self, grid_frequency: int):
        if self.latent_points == 1:
            return [(0.0, 0.0)]
        delta_phi = self.phi_limits[1] - self.phi_limits[0]
        return [(self.phi_limits[0] + point * delta_phi / (grid_frequency - 1), self.theta)
                for point in range(self.latent_points)]


class NearestNeighborsInterpolator:
    def __init__(self,
                 interpolating_indexes: list[int],
                 base_vertices: list[tuple[float, float]],
                 extended_vertices: list[tuple[float, float]],
                 k: int):
        """
        Initialize the interpolator with the base mesh vertices and extended mesh vertices.
        Uses a BallTree for efficient nearest neighbor search.
        """
        self.k = k

        tree = BallTree(self._to_lat_long(base_vertices), metric="haversine")
        distances, indexes = tree.query(self._to_lat_long(extended_vertices), k=self.k)

        distances = distances * 6371000
        clipped = np.clip(distances, a_min=1e-8, a_max=None) ** 2
        inv_distances = torch.tensor(1.0 / clipped, dtype=torch.float32)

        # Normalize weights
        self.weights = inv_distances / inv_distances.sum(dim=-1, keepdim=True)

        self.interp_indexes = torch.as_tensor(interpolating_indexes)
        self.indexes = torch.as_tensor(indexes)
        self.extended_size = extended_vertices.shape[0]

    def _to_lat_long(self, array: list[tuple[float, float]]):
        array = np.array(array)[:, ::-1]  # theta, phi
        array[:, 0] = np.pi/2 - array[:, 0]
        return array

    def __call__(self, f_values: torch.Tensor) -> torch.Tensor:
        """
        Interpolate values at extended points using inverse distance weighting.
        :param f_values: Tensor of shape (..., N), where N is the number of base vertices.
        :return: Interpolated values of shape (..., M), where M is the number of extended vertices.
        """
        shape = f_values.shape
        f_extended = torch.zeros((*shape[:-1], self.extended_size), dtype=f_values.dtype)
        mapped_indexes = self.interp_indexes[self.indexes]

        for idx in range(self.k):
            f_extended += f_values[..., mapped_indexes[..., idx]] * self.weights[..., idx]
        return f_extended

class MeshProcessorBase:
    def __init__(self, init_grid_frequency, phi_limits, boundaries_cond):
        self.init_grid_frequency = init_grid_frequency
        self.phi_limits = phi_limits
        self.boundaries_cond = boundaries_cond
        self.last_point = boundaries_cond != "periodic"

    def _create_theta_lines(self, grid_frequency):
        return [
            ThetaLine(
                theta=(np.pi / 2) * point / grid_frequency,
                points=point,
                phi_limits=self.phi_limits,
                last_point=self.last_point,
                boundaries_cond=self.boundaries_cond
            ) for point in range(1, grid_frequency + 1)
        ]

    def _assemble_vertices(self, theta_lines, vertex_method):
        return np.concatenate(
            [np.array(getattr(tl, vertex_method)(), dtype=np.float32) for tl in theta_lines],
            axis=0
        )

    def _get_post_indexes(self, theta_lines):
        indexes = []
        for tl in theta_lines:
            indexes.extend(self._get_absolute_post_indexes(tl))
        return indexes

    def _get_absolute_post_indexes(self, theta_line):
        return self._get_absolute_indexes(theta_line, "post_indexes")

    def _get_absolute_indexes(self, theta_line, method):
        visible_points = theta_line.visible_points
        if self.last_point:
            shift = (visible_points - 1) * visible_points // 2
        else:
            shift = 2 + ((visible_points - 2) * (visible_points + 1)) // 2 if theta_line.latent_points != 1 else 0
        indexes = getattr(theta_line, method)()
        return [idx + shift for idx in indexes]

    def _triangulate(self, grid_frequency):
        delta_phi = self.phi_limits[1] - self.phi_limits[0]
        vertices = np.array(
            [[0.0, 0.0]] + [
                (self.phi_limits[0] + delta_phi * (q / grid_frequency),
                 (np.pi / 2) * (k / (grid_frequency - 1)))
                for k in range(1, grid_frequency) for q in range(k + 1)
            ]
        )
        triangulation = Delaunay(vertices)
        return triangulation.simplices



class InterpolatingMeshProcessor(MeshProcessorBase):
    def __init__(self, interpolate_grid_frequency, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interpolate_grid_frequency = interpolate_grid_frequency

        self.base_theta_lines = self._create_theta_lines(self.init_grid_frequency)
        self.interpolate_theta_lines = self._create_theta_lines(self.interpolate_grid_frequency)

        self.post_indexes = torch.tensor(self._get_post_indexes(self.base_theta_lines))
        self.final_vertices, self.simplices = self._get_post_mesh()
        self.interpolator = self._get_interpolator(self.final_vertices)

        self.init_vertices = self._assemble_vertices(self.base_theta_lines, "init_phi_theta")
        self.extended_size = self.final_vertices.shape[0]

    def _get_post_mesh(self):
        extended_vertices = self._assemble_vertices(self.interpolate_theta_lines, "post_phi_theta")
        simplices = self._triangulate(self.interpolate_grid_frequency)
        return extended_vertices, simplices

    def _get_interpolating_indexes(self, theta_lines):
        indexes = []
        for tl in theta_lines:
            indexes.extend(self._get_absolute_indexes(tl, "interpolating_indexes"))
        return indexes

    def _get_interpolating_phi_theta(self, theta_lines):
        phi_theta = [pt for tl in theta_lines for pt in tl.interpolating_phi_theta()]
        indexes = self._get_interpolating_indexes(theta_lines)
        return indexes, phi_theta

    def _get_interpolator(self, extended_vertices):
        indexes, phi_theta = self._get_interpolating_phi_theta(self.base_theta_lines)
        return NearestNeighborsInterpolator(indexes, phi_theta, extended_vertices, k=8)

    def post_process(self, f_values: torch.Tensor):
        return self.interpolator(f_values)


class BoundaryMeshProcessor(MeshProcessorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_theta_lines = self._create_theta_lines(self.init_grid_frequency)
        self.post_indexes = torch.tensor(self._get_post_indexes(self.base_theta_lines))
        self.final_vertices, self.simplices = self._get_post_mesh()

        self.init_vertices = self._assemble_vertices(self.base_theta_lines, "init_phi_theta")
        self.extended_size = self.final_vertices.shape[0]

    def _get_post_mesh(self):
        vertices = self._assemble_vertices(self.base_theta_lines, "post_phi_theta")
        simplices = self._triangulate(self.init_grid_frequency)
        return vertices, simplices

    def post_process(self, f_values: torch.Tensor):
        return f_values[..., self.post_indexes]


class SkipMeshProcessor(MeshProcessorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_theta_lines = self._create_theta_lines(self.init_grid_frequency)
        self.final_vertices, self.simplices = self._get_post_mesh()

        self.init_vertices = self._assemble_vertices(self.base_theta_lines, "init_phi_theta")
        self.extended_size = self.final_vertices.shape[0]

    def _get_post_mesh(self):
        vertices = self._assemble_vertices(self.base_theta_lines, "post_phi_theta")
        simplices = self._triangulate(self.init_grid_frequency)
        return vertices, simplices

    def post_process(self, f_values: torch.Tensor):
        return f_values


def mesh_processor_factory(init_grid_frequency,
                           interpolate_grid_frequency,
                           interpolate=False,
                           boundaries_cond=None,
                           phi_limits=(0, 2 * math.pi)):

    if interpolate:
        return InterpolatingMeshProcessor(
            interpolate_grid_frequency,
            init_grid_frequency,
            phi_limits,
            boundaries_cond
        )
    elif boundaries_cond != "periodic":
        return BoundaryMeshProcessor(
            init_grid_frequency,
            phi_limits,
            boundaries_cond
        )
    else:
        return SkipMeshProcessor(
            init_grid_frequency,
            phi_limits,
            boundaries_cond
        )


class DelaunayMeshNeighbour(BaseMesh):
    """Delaunay triangulation-based spherical mesh implementation."""
    """It uses CloughTocher2DInterpolator to interpolate Data"""
    def __init__(self,
                 eps: float = 1e-7,
                 phi_limits: tuple[float, float] = (0, 2 * math.pi),
                 initial_grid_frequency: int = 20,
                 interpolation_grid_frequency: int = 40,
                 boundaries_cond=None,
                 interpolate=True):
        """
        Initialize Delaunay mesh parameters.

        Args:
            eps: Small epsilon value for numerical stability
            phi_limits: Maximum value for phi coordinate (default: full circle)
            initial_grid_frequency: Resolution of initial grid
            interpolation_grid_frequency: Resolution of interpolation grid
        """
        super().__init__()
        self.eps = eps
        self.phi_limit = phi_limits
        self.initial_grid_frequency = initial_grid_frequency
        self.interpolation_grid_frequency = interpolation_grid_frequency
        self.mesh_processor = mesh_processor_factory(initial_grid_frequency, interpolation_grid_frequency,
                                                     phi_limits=phi_limits, interpolate=interpolate,
                                                     boundaries_cond=boundaries_cond)

        (self._initial_grid,
         self._post_grid,
         self._post_simplices) = self.create_initial_cache_data()

    def create_initial_cache_data(self) -> tuple:
        """Create and cache initial mesh data structures."""
        return (
            torch.as_tensor(self.mesh_processor.init_vertices, dtype=torch.float32),
            torch.as_tensor(self.mesh_processor.final_vertices, dtype=torch.float32),
            torch.as_tensor(self.mesh_processor.simplices)
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

    def post_process(self,
                    f_init: torch.Tensor) -> torch.Tensor:
        """
        Format interpolated values for Delaunay representation.

        Args:
            f_init: Interpolated function values

        Returns:
            torch.Tensor: Values formatted for Delaunay triangulation
        """
        return self.mesh_processor.post_process(f_init)
