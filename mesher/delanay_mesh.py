import numpy as np
import torch
import scipy
from scipy.spatial import Delaunay

from .general_mesh import BaseMesh


class MeshProcessor:
    def __init__(self, init_grid_frequency, interpolating_grid_frequency,
                 phi_limit, interpolate):
        self.phi_limit = phi_limit
        self.interpolate = interpolate
        self.init_grid_frequency = init_grid_frequency
        self.interpolating_grid_frequency = interpolating_grid_frequency
        self.theta_eps = np.pi / (2 * max(init_grid_frequency, interpolating_grid_frequency))
        self.base_vertices, self.extended_vertices, self.idx_mask = self._preprocess()

        self.base_tri = self._triangulate(self.base_vertices)


        self.extended_tri = self._triangulate(self.extended_vertices)


        if self.interpolate:
            self.final_vertices = self._create_interpolated_vertices(
                grid_frequency=interpolating_grid_frequency,
                phi_limit=self.phi_limit
            )
            self.final_tri = self._triangulate(self.final_vertices)
        else:
            self.final_vertices = self.extended_vertices
            self.final_tri = self.extended_tri


    def _triangulate(self, vertices: np.ndarray) -> Delaunay:
        """Perform Delaunay triangulation on given vertices."""
        return Delaunay(vertices)


    def _preprocess(self):
        """Prepares base and extended vertices along with interpolation masks."""
        base_vertices, extended_vertices = self._create_base_vertices(
            self.init_grid_frequency, self.phi_limit
        )
        if self.interpolate:
            phi_plus, phi_minus, theta_extended = self._create_interpolating_vertices(
                self.init_grid_frequency, self.phi_limit
            )
            interpolating_vertices = np.concatenate(
                [extended_vertices, phi_plus, phi_minus, theta_extended]
            )
            mask_data = self._compute_masks(
                base_vertices, extended_vertices,
                phi_plus, phi_minus, theta_extended,
                interpolating_vertices
            )
            return base_vertices, interpolating_vertices, mask_data
        else:
            zero_mask = self._get_zero_phi_mask(base_vertices)
            return base_vertices, extended_vertices, ((len(base_vertices), len(extended_vertices), zero_mask),)

    def _get_zero_phi_mask(self, vertices):
        """Creates mask for vertices at phi=0 (excluding origin)."""
        return (vertices[:, 0] == 0.0) & (vertices[:, 1] != 0.0)

    def _create_base_vertices(self, grid_frequency, phi_limit):
        """Generates base grid vertices and extended vertices up to phi_limit."""
        initial_points = np.array([[0.0, 0.0], [phi_limit, self.theta_eps]])

        k_values = np.arange(1, grid_frequency)
        q_per_k = [np.arange(k) for k in k_values]
        q_values = np.concatenate(q_per_k)
        k_repeated = np.repeat(k_values, k_values)

        phi = phi_limit * q_values / k_repeated
        theta = (np.pi / 2) * k_repeated / (grid_frequency - 1)
        #theta = np.arcsin(k_repeated / (grid_frequency - 1))
        grid_points = np.column_stack((phi, theta))

        base_vertices = np.vstack((initial_points, grid_points))

        theta_2pi = (np.pi / 2) * k_values / (grid_frequency - 1)
        #theta_2pi = np.arcsin(k_values / (grid_frequency - 1))

        additional_points = np.column_stack((np.full_like(theta_2pi, phi_limit), theta_2pi))
        extended_vertices = np.vstack((base_vertices, additional_points))

        return base_vertices, extended_vertices

    def _create_interpolating_vertices(self, grid_frequency, phi_limit):
        """Generates vertices for interpolation beyond the base grid."""
        k = np.arange(1, grid_frequency)
        phi_plus = phi_limit * (k + 1) / k
        theta_plus = (np.pi / 2) * k / (grid_frequency - 1)
        #theta_plus = np.arcsin(k / (grid_frequency - 1))

        phi_plus_vertices = np.column_stack((phi_plus, theta_plus))

        phi_minus = -phi_limit / k
        phi_minus_vertices = np.column_stack((phi_minus, theta_plus))

        theta_extended = (np.pi / 2) * grid_frequency / (grid_frequency - 1)
        #theta_extended = (np.pi / 2) + np.arcsin(1 / (grid_frequency - 1))

        q = np.arange(grid_frequency + 1)
        if grid_frequency >= 2:
            phi_extended = phi_limit * (q - 1) / (grid_frequency - 2)
        else:
            phi_extended = np.zeros_like(q)
        theta_extended_vertices = np.column_stack((phi_extended, np.full_like(phi_extended, theta_extended)))

        return phi_plus_vertices, phi_minus_vertices, theta_extended_vertices

    def _create_interpolated_vertices(self, grid_frequency, phi_limit):
        """Creates vertices for the interpolated grid."""
        k_values = np.arange(1, grid_frequency)
        q_per_k = [np.arange(k + 1) for k in k_values]
        q_values = np.concatenate(q_per_k)
        k_repeated = np.repeat(k_values, k_values + 1)

        phi = phi_limit * q_values / np.maximum(k_repeated, 1)  # Avoid division by zero

        theta = (np.pi / 2) * k_repeated / (grid_frequency - 1)
        #theta = (np.pi / 2) * np.arcsin(k_repeated / (grid_frequency - 1))
        interpolated_points = np.column_stack((phi, theta))

        initial_points = np.array([[0.0, 0.0], [phi_limit, self.theta_eps]])
        return np.vstack((initial_points, interpolated_points))

    def _compute_masks(self, base_vertices, extended_vertices, phi_plus,
                      phi_minus, theta_extended, interpolating_vertices):
        """Computes masks for function value extension using spatial queries."""
        zero_mask = self._get_zero_phi_mask(base_vertices)
        tree = scipy.spatial.cKDTree(interpolating_vertices)

        # Mask for phi_plus vertices (wrapped by -phi_limit)
        query_points = np.column_stack((phi_plus[:, 0] - self.phi_limit, phi_plus[:, 1]))
        _, indices = tree.query(query_points, distance_upper_bound=1e-6)
        valid = (indices >= 0) & (indices < len(interpolating_vertices))
        mask_phi_plus = np.isin(np.arange(len(interpolating_vertices)), indices[valid])

        # Mask for phi_minus vertices (wrapped by +phi_limit)
        query_points = np.column_stack((phi_minus[:, 0] + self.phi_limit, phi_minus[:, 1]))
        _, indices = tree.query(query_points, distance_upper_bound=1e-6)
        valid = (indices >= 0) & (indices < len(interpolating_vertices))
        mask_phi_minus = np.isin(np.arange(len(interpolating_vertices)), indices[valid])

        # Mask for theta_extended vertices (mirrored in theta)
        query_points = np.column_stack((theta_extended[:, 0], np.pi - theta_extended[:, 1]))
        _, indices = tree.query(query_points, distance_upper_bound=1e-6)
        valid = (indices >= 0) & (indices < len(interpolating_vertices))
        mask_theta_ext = np.isin(np.arange(len(interpolating_vertices)), indices[valid])

        return (
              (len(base_vertices), len(extended_vertices), zero_mask),
              (len(extended_vertices), len(extended_vertices)+len(phi_plus), mask_phi_plus),
              (len(extended_vertices)+len(phi_plus), len(extended_vertices)+len(phi_plus)+len(phi_minus), mask_phi_minus),
              (len(extended_vertices)+len(phi_plus)+len(phi_minus), len(interpolating_vertices), mask_theta_ext)
        )

    def _get_extended_function(self, f_values):
        shape = f_values.shape
        f_extended = np.zeros((*shape[:-1], len(self.extended_vertices[:, 0])))
        f_extended[..., :len(self.base_vertices)] = f_values
        begin, end, zero_mask = self.idx_mask[0]
        f_extended[..., begin:end] = f_values[..., zero_mask]
        if self.interpolate:
            for begin, end, mask in self.idx_mask[1:]:
                f_extended[..., begin:end] = f_extended[..., mask]
            else:
                pass
        return f_extended

    def post_process(self, f_values):
        f_extended = self._get_extended_function(f_values)
        if self.interpolate:
            interpolator = scipy.interpolate.CloughTocher2DInterpolator(self.extended_tri,
                                                                        f_extended.swapaxes(-1, 0)
                                                                        )
            return interpolator(self.final_tri.points).swapaxes(-1, 0)
        else:
            return f_extended

# It must be changed in the future. Now the mesh is not spherical but plat. Use ConvexHull / matplotlib.tri or
# other libraries
# Reimplement approximation processs
class DelaunayMeshClough(BaseMesh):
    """Delaunay triangulation-based spherical mesh implementation."""
    """It uses CloughTocher2DInterpolator to interpolate Data"""
    def __init__(self,
                 eps: float = 1e-7,
                 phi_limit: float = 2 * np.pi,
                 initial_grid_frequency: int = 20,
                     interpolation_grid_frequency: int = 40,
                 interpolate=True):
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
        self.mesh_processor = MeshProcessor(initial_grid_frequency, interpolation_grid_frequency,
                                            phi_limit=phi_limit, interpolate=interpolate)

        (self._initial_grid,
         self._initial_simplices,
         self._post_grid,
         self._post_simplices) = self.create_initial_cache_data()

    def create_initial_cache_data(self) -> tuple:
        """Create and cache initial mesh data structures."""
        return (
            torch.as_tensor(self.mesh_processor.base_vertices, dtype=torch.float32),
            torch.as_tensor(self.mesh_processor.base_tri.simplices),
            torch.as_tensor(self.mesh_processor.final_vertices, dtype=torch.float32),
            torch.as_tensor(self.mesh_processor.final_tri.simplices)
        )

    def _triangulate(self, vertices: np.ndarray) -> Delaunay:
        """Perform Delaunay triangulation on given vertices."""
        return Delaunay(vertices)

    @property
    def initial_mesh(self):
        return self._initial_grid, self._initial_simplices

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
            f_interpolated: Interpolated function values
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
            simplices: Simplices to use for final representation

        Returns:
            torch.Tensor: Values formatted for Delaunay triangulation
        """
        #return f_init
        f_values = f_init.numpy()
        return torch.as_tensor(self.mesh_processor.post_process(f_values))



