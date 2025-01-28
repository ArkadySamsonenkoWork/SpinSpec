import functools

import torch

import constants
import particles
import res_field
import utils


def kronecker_product(matrices: list) -> torch.Tensor:
    """Computes the Kronecker product of a list of matrices."""
    return functools.reduce(torch.kron, matrices)


def create_operator(system_particles: list, target_idx: int, matrix: torch.Tensor) -> torch.Tensor:
    """Creates an operator acting on the target particle with identity elsewhere."""
    operator = []
    for i, p in enumerate(system_particles):
        operator.append(matrix if i == target_idx else p.identity)
    return kronecker_product(operator)


def scalar_tensor_multiplication(
    tensor_components_A: torch.Tensor,
    tensor_components_B: torch.Tensor,
    transformation_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Computes the scalar product of two tensor components after applying a transformation.
    Parameters:
        tensor_components_A (torch.Tensor): Input tensor components with shape [..., 3, K, K].
        tensor_components_B (torch.Tensor): Input tensor components with shape [..., 3, K, K].
        transformation_matrix (torch.Tensor): Transformation matrix with shape [..., 3, 3].
    Returns:
        torch.Tensor: Scalar product with shape [..., K, K].
    """
    return torch.einsum(
        '...ij,...jkl,...ikl->...kl',
        transformation_matrix,
        tensor_components_A,
        tensor_components_B
    )


def transform_tensor_components(tensor_components: torch.Tensor, transformation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Applies a matrix transformation to a collection of tensor components.
    Parameters:
        tensor_components (torch.Tensor): Input tensor components with shape [..., 3, K, K]
            where 3 represents different components (e.g., x,y,z) and K is the system dimension. For example,
            [Sx, Sy, Sz]
        transformation_matrix (torch.Tensor): Transformation matrix with shape [..., 3, 3]. For example, g - tensor
    Returns:
        torch.Tensor: Transformed tensor components with shape [..., 3, K, K]
    """
    return torch.einsum('...ij,...jkl->...ikl', transformation_matrix, tensor_components)


class SpinSystem:
    """Represents a spin system with electrons, nuclei, and interactions."""
    def __init__(self, particles: list[particles.Nucleus], electron_indices: list[int],
                 g_tensors: list[torch.Tensor],
                 hyperfine_interactions: list | None = None,
                 device=torch.device("cpu")):
        self.particles = particles
        self.electron_indices = electron_indices
        self.nucleus_indices = [i for i in range(len(self.particles)) if i not in electron_indices]
        self.g_tensors = [g.to(torch.complex64) for g in g_tensors]  # Ensure complex type
        self.hyperfine_interactions = hyperfine_interactions if hyperfine_interactions else []

        self._operator_cache = {}  # Format: {particle_idx: tensor}
        self._precompute_all_operators()

        self.device = device
        self.batch_shape = self._get_batch_shape(self.g_tensors[0])

    def _get_batch_shape(self, tensor_component):
        shape = tensor_component.shape
        if len(shape) > 2:
            return shape[:-2]
        return (1, )

    @property
    def dim(self) -> int:
        """Dimension of the system's Hilbert space."""
        return torch.prod(torch.tensor([int(2 * p.spin + 1) for p in self.particles])).item()

    def _precompute_all_operators(self):
        """Precompute spin operators for all particles in the full Hilbert space."""
        for idx, p in enumerate(self.particles):
            axis_cache = []
            for axis, mat in zip(['x', 'y', 'z'], p.spin_matrices):
                operator = create_operator(self.particles, idx, mat)
                axis_cache.append(operator)
            self._operator_cache[idx] = torch.stack(axis_cache, dim=-3)

    def build_zero_field_term(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        F = torch.zeros((*self.batch_shape, self.dim, self.dim), dtype=torch.complex64, device=self.device)
        for e_idx, n_idx, A in self.hyperfine_interactions:
            A = A.to(torch.complex64)
            F += scalar_tensor_multiplication(self._operator_cache[e_idx], self._operator_cache[n_idx], A)
        F = F
        return F

    def _build_electron_zeeman_terms(self) -> torch.Tensor:
        """Constructs the Zeeman interaction terms Gx, Gy, Gz. for electron spins with give g-tensors"""
        G = torch.zeros((*self.batch_shape, 3, self.dim, self.dim), dtype=torch.complex64, device=self.device)
        for idx, e_idx in enumerate(self.electron_indices):
            g = self.g_tensors[idx]
            G += transform_tensor_components(self._operator_cache[idx], g)
        G *= (constants.BOHR / constants.PLANCK)
        return G

    def _build_nucleus_zeeman_terms(self) -> torch.Tensor:
        """Constructs the Nucleus interaction terms Gx, Gy, Gz. for nucleus spins"""
        G = torch.zeros((*self.batch_shape, 3, self.dim, self.dim), dtype=torch.complex64, device=self.device)
        for idx, n_idx in enumerate(self.nucleus_indices):
            g = self.particles[n_idx].g_factor
            G += self._operator_cache[idx] * g
        G *= (constants.NUCLEAR_MAGNETRON / constants.PLANCK)
        return G

    def build_zeeman_terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Constructs the Zeeman interaction terms Gx, Gy, Gz. for the system"""
        G = self._build_electron_zeeman_terms() + self._build_nucleus_zeeman_terms()
        return G[..., 0, :, :], G[..., 1, :, :], G[..., 2, :, :]

    def calculate_derivative_max(self):
        """
        Calculate the maximum value of the energy derivatives with respect to magnetic field.
        It is assumed that B has direction along z-axis
        :return: the maximum value of the energy derivatives with respect to magnetic field
        """
        electron_contrib = 0
        for idx, e_idx in enumerate(self.electron_indices):
            electron_contrib += self.particles[e_idx].spin * torch.sum(
                self.g_tensors[idx][..., :, 0], dim=-1, keepdim=True).abs()

        nuclei_contrib = 0
        for idx, n_idx in enumerate(self.nucleus_indices):
            p = self.particles[n_idx]
            nuclei_contrib += p.spin * p.g_factor.abs()

        return electron_contrib * (constants.BOHR / constants.PLANCK) +\
            nuclei_contrib * (constants.NUCLEAR_MAGNETRON / constants.PLANCK)

    def get_hamiltonian_terms(self) -> tuple:
        """
        Returns F, Gx, Gy, Gz.
        F is magnetic field free term
        Gx, Gy, Gz are terms multiplied to Bx, By, Bz respectively

        """
        return self.build_zero_field_term(), *self.build_zeeman_terms()

def system_1():
    electron = particles.Electron(spin=1/2)
    g_isotropic = torch.diag(torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32))

    system = SpinSystem(
        particles=[electron],
        electron_indices=[0],
        g_tensors=[g_isotropic]
    )
    return system

def system_3():
    electron = particles.Electron(spin=1/2)
    g_isotropic = torch.diag(torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32))
    g_tensor = torch.stack((g_isotropic, g_isotropic, g_isotropic))

    system = SpinSystem(
        particles=[electron],
        electron_indices=[0],
        g_tensors=[g_tensor]
    )
    return system


def system_2():
    electron = particles.Electron(spin=1 / 2)
    g_isotropic = torch.diag(torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32))
    nucleus = particles.Nucleus("14N")
    A = torch.eye(3, dtype=torch.float32) * 5 * 10**10

    system = SpinSystem(
        particles=[electron, nucleus],
        electron_indices=[0],
        g_tensors=[g_isotropic],
        hyperfine_interactions=[(0, 1, A)]
    )
    return system

if __name__ == "__main__":
    resonance_frequency = torch.tensor([9.8 * 1e9])
    B = 0.3  # T
    B_high = torch.tensor([100.0])
    B_low = torch.tensor([0.2])
    system = system_2()
    F, Gx, Gy, Gz = system.get_hamiltonian_terms()
    deriv_max = system.calculate_derivative_max()[..., None]
    res_field.get_resonance_intervals(F=F, Gz=Gz, B_low=B_low, B_high=B_high, deriv_max=deriv_max,
                                            resonance_frequency=resonance_frequency)
