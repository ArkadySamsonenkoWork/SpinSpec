import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.spatial import Delaunay


# scipy.ndimage.map_coordinates could be usefull
class BaseMesh(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @property
    def initial_size(self):
        return self.initial_grid.size()[:-1]

    def create_rotation_matrices(self):
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

    @staticmethod
    def spherical_triangle_areas(vertices, triangles):
        """
        vertices: tensor of shape (N,2), each row is [phi, theta]
        triangles: tensor of shape (M,3) with indices into vertices defining the triangles.

        Returns:
           areas: tensor of shape (M,) with the spherical areas of the triangles (for unit sphere).
                  For a sphere of radius R, multiply each area by R**2.
        """

        def _angle_between(u, v):
            dot = (u * v).sum(dim=1)
            # Clamp the dot product to avoid numerical errors outside [-1, 1]
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


        # It is so-called Spherical law of cosines
        alpha = torch.acos(
            torch.clamp((torch.cos(a) - torch.cos(b) * torch.cos(c)) / (torch.sin(b) * torch.sin(c)), -1.0, 1.0))
        beta = torch.acos(
            torch.clamp((torch.cos(b) - torch.cos(c) * torch.cos(a)) / (torch.sin(c) * torch.sin(a)), -1.0, 1.0))
        gamma = torch.acos(
            torch.clamp((torch.cos(c) - torch.cos(a) * torch.cos(b)) / (torch.sin(a) * torch.sin(b)), -1.0, 1.0))
        excess = (alpha + beta + gamma) - torch.pi
        excess = torch.nan_to_num(excess, 0.0)  # НАДО РАЗОБРАТЬСЯ, ПОЧЕМУ ЕСТЬ БИТЫЕ УЧАСТКИ ПЛОЩАДИ
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


# Пока реализована линейная интерполяция.
# Как вариант - сделать интерполяцию по phi - кубическую. В таком случае удобно ввести граничные условия
# Потом сделать интерполяцию по тета на оценках
class DelaunayMeshLinear(BaseMesh):
    """Delaunay triangulation-based spherical mesh implementation."""

    def __init__(self,
                 eps: float = 1e-7,
                 phi_limit: float = 2 * np.pi,
                 initial_grid_frequency: int = 40,
                 interpolation_grid_frequency: int = 20):
        """
        Initialize Delaunay mesh parameters.

        Args:
            eps: Small epsilon value for numerical stability
            phi_limit: Maximum value for phi coordinate (default: full circle)
            initial_grid_frequency: Resolution of initial grid
            interpolation_grid_frequency: Resolution of interpolation grid
        """
        super().__init__()
        self.eps = eps
        self.phi_limit = phi_limit
        self.initial_grid_frequency = initial_grid_frequency
        self.interpolation_grid_frequency = interpolation_grid_frequency

        # Initialize mesh data structures
        (self._initial_grid,
         self._initial_simplices,
         self._interpolation_grid,
         self._interpolation_indices,
         self._barycentric_coords,
         self._interpolation_simplices) = self.create_initial_cache_data()

    def create_initial_cache_data(self) -> tuple:
        """Create and cache initial mesh data structures."""
        initial_grid = self.create_grid(self.initial_grid_frequency, self.phi_limit)
        initial_tri = self._triangulate(initial_grid)

        interpolation_grid = self.create_grid(self.interpolation_grid_frequency, self.phi_limit)
        interpolation_tri = self._triangulate(interpolation_grid)

        # Convert to tensors
        return (
            torch.as_tensor(initial_grid),
            torch.as_tensor(initial_tri.simplices),
            torch.as_tensor(interpolation_grid),
            *self.get_interpolation_coefficients(interpolation_grid, initial_tri),
            torch.as_tensor(interpolation_tri.simplices)
        )

    def _triangulate(self, vertices: np.ndarray) -> Delaunay:
        """Perform Delaunay triangulation on given vertices."""
        return Delaunay(vertices)

    @property
    def initial_mesh(self):
        return self._initial_grid, self._initial_simplices

    @property
    def post_mesh(self):
        return self._interpolation_grid, self._interpolation_simplices

    @property
    def initial_grid(self):
        return self._initial_grid


    def create_grid(self, grid_frequency: int = 10, phi_limit: float = 2 * np.pi):
        """
        Create grid points using the parameterization:
        K == grid_frequency
           theta = (pi/2)*(k/K - 1)   for k = 0,1,...,K      (θ in [-pi/2, 0])
           phi   = (pi/2)*(q/k)       for q = 0,1,...,k
        For k == 0, we only have one point (q==0).
        """
        self.phi_limit = phi_limit
        #vertices = [(0.0, 0.0), (phi_limit, 0.001), (0.0, 0.005), (phi_limit, 0.005)] +\
        #           [(phi_limit * (q / k), (np.pi / 2) * (k / (grid_frequency - 1)))
        #            for k in range(1, grid_frequency) for q in range(k+1)]
        vertices = [(phi_limit * (q / k), (np.pi / 2) * (k / (grid_frequency - 1)))
                    for k in range(1, grid_frequency) for q in range(k+1)]

        #N = 20
        #M = 40
        #phi_ar = [2 * self.phi_limit * n / (N-1) for n in range(N)]
        #theta_ar = [(np.pi / 2) * m / (M-1) for m in range(M)]
        #vertices = [(phi, theta) for phi in phi_ar for theta in theta_ar]
        return vertices

    def reflect_coordinate(self, x: torch.Tensor,
                           lower: float, upper: float) -> torch.Tensor:
        """
        Reflect coordinates into [lower, upper] interval.

        Args:
            x: Input tensor
            lower: Interval lower bound
            upper: Interval upper bound

        Returns:
            torch.Tensor: Reflected coordinates
        """
        interval_length = upper - lower
        x_shifted = x - lower
        x_mod = torch.remainder(x_shifted, 2 * interval_length)
        return lower + torch.where(x_mod > interval_length,
                                   2 * interval_length - x_mod,
                                   x_mod)

    def get_interpolation_coefficients(self,
                                       query_points: np.ndarray,
                                       triangulation: Delaunay) -> tuple:
        """
        Compute barycentric interpolation coefficients for query points.

        Args:
            query_points: Points to interpolate (N, 2)
            triangulation: Delaunay triangulation of original grid

        Returns:
            tuple: (simplex indices, barycentric coordinates)
        """
        query_tensor = torch.as_tensor(query_points, dtype=torch.float32)

        # Reflect coordinates into valid range
        query_tensor[:, 0] = self.reflect_coordinate(query_tensor[:, 0],
                                                     0.0, self.phi_limit)
        query_tensor[:, 1] = self.reflect_coordinate(query_tensor[:, 1],
                                                     0.0, np.pi / 2)

        # Find containing simplices
        simplex_indices = self.find_containing_simplices(
            query_tensor.numpy(), triangulation)

        # Compute barycentric coordinates
        transforms = triangulation.transform[simplex_indices, :2, :]
        offsets = query_tensor.numpy() - triangulation.transform[simplex_indices, 2, :]
        bary = np.einsum('nij,nj->ni', transforms, offsets)
        bary_coords = np.hstack([bary, 1 - bary.sum(axis=1, keepdims=True)])

        return triangulation.simplices[simplex_indices], bary_coords

    def find_containing_simplices(self,
                                  query_points: np.ndarray,
                                  triangulation: Delaunay) -> np.ndarray:
        """
        Find simplices containing query points with boundary handling.

        Args:
            query_points: Points to locate (N, 2)
            triangulation: Delaunay triangulation to search

        Returns:
            np.ndarray: Indices of containing simplices
        """
        adjusted_points = query_points.copy()

        # Handle boundary points by small perturbations
        boundaries = {
            'phi_low': (adjusted_points[:, 0] == 0, self.eps),
            'phi_high': (adjusted_points[:, 0] == self.phi_limit, -self.eps),
            'theta_low': (adjusted_points[:, 1] == 0, self.eps),
            'theta_high': (adjusted_points[:, 1] == np.pi / 2, -self.eps)
        }

        for mask, delta in boundaries.values():
            adjusted_points[mask] += delta

        return triangulation.find_simplex(adjusted_points)

    def interpolate(self, f_values: torch.Tensor) -> torch.Tensor:
        """
        Interpolate function values using barycentric coordinates.

        Args:
            f_values: Function values at initial grid points

        Returns:
            torch.Tensor: Interpolated values at query points
        """
        raise NotImplementedError
        return (f_values[..., self._interpolation_indices] *
                self._barycentric_coords).sum(dim=-1)

    def to_delaunay(self,
                    f_interpolated: torch.Tensor,
                    simplices: torch.Tensor) -> torch.Tensor:
        """
        Format interpolated values for Delaunay representation.

        Args:
            f_interpolated: Interpolated function values
            simplices: Simplices to use for final representation

        Returns:
            torch.Tensor: Values formatted for Delaunay triangulation
        """
        return f_interpolated[..., simplices]

    @abstractmethod
    def post_process(self, f_function: torch.Tensor):
        pass

#### IT IS BETTER TO USE CUPY AND SCIPY RESPECTEVELY
class QuadraticMesh(BaseMesh):
    """Delaunay squared-based spherical mesh implementation."""

    def __init__(self,
                 eps: float = 1e-7,
                 phi_limit: float = 2 * np.pi,
                 initial_theta_frequency: int = 30,
                 initial_phi_frequency: int = 30,
                 interpolation_theta_frequency: int = 50,
                 interpolation_phi_frequency: int = 6):
        """
        Initialize Delaunay mesh parameters.

        Args:
            eps: Small epsilon value for numerical stability
            phi_limit: Maximum value for phi coordinate (default: full circle)
            initial_theta_frequency: Resolution of initial theta grid
            initial_phi_frequency: Resolution of initial phi grid
            interpolation_theta_frequency: Resolution of interpolation theta grid
            interpolation_phi_frequency: Resolution of interpolation phi grid
        """
        super().__init__()
        self.eps = eps
        self.phi_limit = phi_limit
        self.initial_theta_frequency = initial_theta_frequency
        self.initial_phi_frequency = initial_phi_frequency
        self.interpolation_theta_frequency = interpolation_theta_frequency
        self.interpolation_phi_frequency = interpolation_phi_frequency

        # Initialize mesh data structures
        (self._initial_grid,
         self._initial_simplices,
         self._interpolation_grid,
         self._interpolation_indices,
         self._barycentric_coords,
         self._interpolation_simplices) = self.create_initial_cache_data()

    def create_initial_cache_data(self) -> tuple:
        """Create and cache initial mesh data structures."""
        initial_grid = self.create_grid(self.initial_theta_frequency, self.initial_phi_frequency, self.phi_limit)
        initial_tri = self._triangulate(initial_grid)

        interpolation_grid = self.create_grid(self.interpolation_theta_frequency,
                                              self.interpolation_phi_frequency, self.phi_limit)
        interpolation_tri = self._triangulate(interpolation_grid)

        # Convert to tensors
        return (
            torch.as_tensor(initial_grid),
            torch.as_tensor(initial_tri.simplices),
            torch.as_tensor(interpolation_grid),
            *self.get_interpolation_coefficients(interpolation_grid, initial_tri),
            torch.as_tensor(interpolation_tri.simplices)
        )

    def _triangulate(self, vertices: np.ndarray) -> Delaunay:
        """Perform Delaunay triangulation on given vertices."""
        return Delaunay(vertices)

    @property
    def initial_mesh(self):
        return self._initial_grid, self._initial_simplices

    @property
    def post_mesh(self):
        return self._interpolation_grid, self._interpolation_simplices

    @property
    def initial_grid(self):
        return self._initial_grid

    @property
    def size(self):
        return self.grid.size()[:-1]

    def _theta_to_sin_dist(self, theta_frequency):
        u = np.linspace(0, 1, theta_frequency)
        theta = np.arccos(1 - u)
        return theta

    def create_grid(self, theta_frequency: int = 10, phi_frequency: int = 10, phi_limit: float = 2 * np.pi):
        """
        Create grid points using the parameterization:
        K == grid_frequency
           theta = (pi/2)*(k/K - 1)   for k = 0,1,...,K      (θ in [-pi/2, 0])
           phi   = (pi/2)*(q/k)       for q = 0,1,...,k
        For k == 0, we only have one point (q==0).
        """
        phi_ar = [phi_limit * n / (phi_frequency-1) for n in range(phi_frequency)]
        #theta_ar = [phi_limit * m / (theta_frequency-1) for m in range(theta_frequency)]
        theta_ar = self._theta_to_sin_dist(theta_frequency)
        vertices = np.array([(phi, theta) for theta in theta_ar for phi in phi_ar], dtype=np.float32)
        return vertices

    def reflect_coordinate(self, x: torch.Tensor,
                           lower: float, upper: float) -> torch.Tensor:
        """
        Reflect coordinates into [lower, upper] interval.

        Args:
            x: Input tensor
            lower: Interval lower bound
            upper: Interval upper bound

        Returns:
            torch.Tensor: Reflected coordinates
        """
        interval_length = upper - lower
        x_shifted = x - lower
        x_mod = torch.remainder(x_shifted, 2 * interval_length)
        return lower + torch.where(x_mod > interval_length,
                                   2 * interval_length - x_mod,
                                   x_mod)

    def get_interpolation_coefficients(self,
                                       query_points: np.ndarray,
                                       triangulation: Delaunay) -> tuple:
        """
        Compute barycentric interpolation coefficients for query points.

        Args:
            query_points: Points to interpolate (N, 2)
            triangulation: Delaunay triangulation of original grid

        Returns:
            tuple: (simplex indices, barycentric coordinates)
        """
        query_tensor = torch.as_tensor(query_points, dtype=torch.float32)

        # Reflect coordinates into valid range
        query_tensor[:, 0] = self.reflect_coordinate(query_tensor[:, 0],
                                                     0.0, self.phi_limit)
        query_tensor[:, 1] = self.reflect_coordinate(query_tensor[:, 1],
                                                     0.0, np.pi / 2)

        # Find containing simplices
        simplex_indices = self.find_containing_simplices(
            query_tensor.numpy(), triangulation)

        # Compute barycentric coordinates
        transforms = triangulation.transform[simplex_indices, :2, :]
        offsets = query_tensor.numpy() - triangulation.transform[simplex_indices, 2, :]
        bary = np.einsum('nij,nj->ni', transforms, offsets)
        bary_coords = np.hstack([bary, 1 - bary.sum(axis=1, keepdims=True)])

        return triangulation.simplices[simplex_indices], bary_coords

    def find_containing_simplices(self,
                                  query_points: np.ndarray,
                                  triangulation: Delaunay) -> np.ndarray:
        """
        Find simplices containing query points with boundary handling.

        Args:
            query_points: Points to locate (N, 2)
            triangulation: Delaunay triangulation to search

        Returns:
            np.ndarray: Indices of containing simplices
        """
        adjusted_points = query_points.copy()

        # Handle boundary points by small perturbations
        boundaries = {
            'phi_low': (adjusted_points[:, 0] == 0, self.eps),
            'phi_high': (adjusted_points[:, 0] == self.phi_limit, -self.eps),
            'theta_low': (adjusted_points[:, 1] == 0, self.eps),
            'theta_high': (adjusted_points[:, 1] == np.pi / 2, -self.eps)
        }

        for mask, delta in boundaries.values():
            adjusted_points[mask] += delta

        return triangulation.find_simplex(adjusted_points)

    def post_process(self, f):
        return f

    def to_delaunay(self,
                    f_interpolated: torch.Tensor,
                    simplices: torch.Tensor) -> torch.Tensor:
        """
        Format interpolated values for Delaunay representation.

        Args:
            f_interpolated: Interpolated function values
            simplices: Simplices to use for final representation

        Returns:
            torch.Tensor: Values formatted for Delaunay triangulation
        """
        return f_interpolated[..., simplices]

    def interpolate(self, f_values: torch.Tensor) -> torch.Tensor:
        """
        Interpolate function values using barycentric coordinates.

        Args:
            f_values: Function values at initial grid points

        Returns:
            torch.Tensor: Interpolated values at query points
        """
        raise NotImplementedError
        print(f_values.shape)
        batch_shape = f_values.shape[:-1]
        f_2d = f_values.view(*batch_shape, self.initial_theta_frequency, self.initial_phi_frequency)
        f_tensor = f_2d.unsqueeze(0).unsqueeze(0)
        #f_padded = F.pad(f_tensor, pad=(0, 1), mode='constant', value=0)
        #f_padded[..., :, -1] = f_tensor[..., :, 0]  # Now, f_padded shape is [1, 1, theta_frequency, phi_frequency+1]
        # Prepare new grid for interpolation (example new grid dimensions):
        new_phi_frequency = 120
        new_theta_frequency = 60


        phi_ar = [self.phi_limit * n / (self.interpolation_phi_frequency-1) for n in range(self.interpolation_phi_frequency)]
        theta_ar = [(math.pi / 2) * m / (self.interpolation_theta_frequency - 1) for m in
                  range(self.interpolation_theta_frequency)]

        phi_new = torch.as_tensor(phi_ar)
        theta_new = torch.as_tensor(theta_ar)


        #phi_idx = phi_new / self.phi_limit * self.phi_frequency  # fractional index [0, phi_frequency)
        # For theta, indices go from 0 to (theta_frequency - 1)
        #theta_idx = theta_new / (math.pi / 2) * (theta_frequency - 1)

        # Convert indices to normalized coordinates in [-1, 1]:
        # Width (phi) normalized using padded width = phi_frequency
        x_norm = (phi_new / self.phi_limit) * 2 - 1
        # Height (theta) normalized using theta range = (theta_frequency - 1)
        y_norm = (theta_new / (math.pi / 2)) * 2 - 1

        y_grid, x_grid = torch.meshgrid(y_norm, x_norm, indexing='ij')
        grid = torch.stack((x_grid, y_grid), dim=-1).unsqueeze(0)  # [1, new_theta_frequency, new_phi_frequency, 2]
        print(grid.shape)
        # Now, interpolate:
        #f_interp = F.grid_sample(f_2d, grid, mode='bilinear', align_corners=True, padding_mode="border")
        # f_interp now has shape [1, 1, new_theta_frequency, new_phi_frequency]

        # Remove extra dimensions:
        #f_interp = f_interp.squeeze(0).squeeze(0)  # Shape: [new_theta_frequency, new_phi_frequency]
        result = torch.nn.functional.interpolate(f_2d, size=(self.interpolation_theta_frequency, self.interpolation_phi_frequency)).flatten(-2, -1)
        print(result.shape)
        return result

