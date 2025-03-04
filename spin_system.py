import functools
import itertools
from abc import ABC, abstractmethod

import torch

import constants
import particles
import res_field_algorithm
import utils

# Подумать над изменением логики.
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

# Возможно, стоит переделать логику работы расчёта тензоров через тенорное произведение. Сделать отдельный тип данных.
# Сейчас каждый спин даёт матрицу [K, K] и расчёт взаимодействией не оптимальный
class BaseInteraction(ABC):
    @property
    @abstractmethod
    def tensor(self):
        pass

    @property
    @abstractmethod
    def strained_tensor(self):
        pass

    @property
    @abstractmethod
    def batche_shape(self):
        pass


class Interaction(BaseInteraction):
    def __init__(self, components: torch.Tensor,
                 frame: torch.Tensor = None, strain: torch.Tensor = None):
        self.shape = components.shape
        if self.shape[-1] != 3:
            raise ValueError("The components size must be 3. Please point it as x, y, z components")
        self.components = components.to(torch.complex64)  # in main axes. Must be 3 components

        if frame is None:
            self._frame = torch.zeros((*self.shape, 3), dtype=torch.float32)  # alpha, beta, gamma
        elif frame.shape != self.shape:
            raise ValueError("The frame shape must be equal to shape of the initial components."
                             "Please point it as alpha, beta, gamma Euler angles.")
        else:
            self._frame = frame

        if (strain is None) or (strain.shape == self.shape):
            self.strain = strain
            if self.strain is not None:
                self.strain = self.strain.to(torch.complex64)
        else:
            raise ValueError("The strain shape must be equal to shape of the initial components."
                             "Please point it as x, y, z components")

        self._rot_matrix = self.euler_to_rotmat(self._frame)

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        if frame is None:
            self._frame = torch.tensor([0.0, 0.0, 0.0]) # alpha, beta, gamma
        self._rot_matrix = self.euler_to_rotmat(self._frame)


    def euler_to_rotmat(self, euler_angles: torch.Tensor):
        """
        Convert a tensor of Euler angles (alpha, beta, gamma) to a rotation matrix.
        Args:
            euler_angles (torch.Tensor): Tensor of shape [..., 3] containing Euler angles [α, β, γ] in radians.

        Returns:
            torch.Tensor: Rotation matrix of shape (..., 3, 3)
        """
        alpha, beta, gamma = euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]

        ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
        sa, sb, sg = torch.sin(alpha), torch.sin(beta), torch.sin(gamma)

        R = torch.stack([
            torch.stack([ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg], dim=-1),
            torch.stack([sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg], dim=-1),
            torch.stack([-sb, cb * sg, cb * cg], dim=-1)
        ], dim=-2)

        return R.to(torch.complex64)

    def _tensor(self):
        """
        :return: the tensor in the spin system axis
        the shape of the returned tensor is [..., 3, 3]
        """
        return utils.apply_single_rotation(self._rot_matrix, torch.diag_embed(self.components))

    def _strained_tensor(self):
        """
        :return: return the None or the tensor with the shape [..., 3, 3, 3] or None
        """
        if self.strain is None:
            return None
        else:
            return self.strain.unsqueeze(-1).unsqueeze(-1) *\
                torch.einsum("...ik, ...jk->...kij", self._rot_matrix, self._rot_matrix)

    @property
    def tensor(self):
        return self._tensor()

    @property
    def strained_tensor(self):
        return self._strained_tensor()

    @property
    def batche_shape(self):
        return self.shape[:-1]


class MultiOrientedInteraction(BaseInteraction):
    def __init__(self, base_tensor, base_strained_tensor, base_batche_shape):
        self._oriented_tensor = base_tensor
        self._strained_tensor = base_strained_tensor
        self._batche_shape = base_batche_shape

    @property
    def tensor(self):
        return self._oriented_tensor

    @property
    def strained_tensor(self):
        return self._strained_tensor

    @property
    def batche_shape(self):
        return self._batche_shape[:-1]


class SpinSystem:
    """Represents a spin system with electrons, nuclei, and interactions."""
    def __init__(self, electrons: list[particles.Electron], g_tensors: list[BaseInteraction],
                 hyperfine_interactions: list[tuple[int, int, BaseInteraction]] | None = None,
                 nuclei: list[particles.Nucleus] | None = None,
                 device=torch.device("cpu")):
        self.electrons = electrons
        self.g_tensors = g_tensors
        self.nuclei = nuclei if nuclei else []
        self.hyperfine_interactions = hyperfine_interactions if hyperfine_interactions else []
        if len(self.g_tensors) != len(self.electrons):
            raise ValueError("the number of g tensors must be equal to the number of electrons")

        self._operator_cache = {}  # Format: {particle_idx: tensor}
        self._precompute_all_operators()

        self.device = device
        self.batch_shape = self.g_tensors[0].batche_shape

    @property
    def dim(self) -> int:
        """Dimension of the system's Hilbert space."""
        return torch.prod(
            torch.tensor([int(2 * p.spin + 1) for p in itertools.chain(self.electrons, self.nuclei)])
        ).item()

    #  Нужно передалать функцию, слишком много циклов. Можно улучшить
    def _precompute_all_operators(self):
        """Precompute spin operators for all particles in the full Hilbert space."""
        particels = self.electrons + self.nuclei
        for idx, p in enumerate(particels):
            axis_cache = []
            for axis, mat in zip(['x', 'y', 'z'], p.spin_matrices):
                operator = create_operator(particels, idx, mat)
                axis_cache.append(operator)
            self._operator_cache[idx] = torch.stack(axis_cache, dim=-3)  # Сейчас каждый спин даёт матрицу [K, K] и
                                                                         # расчёт взаимодействией не оптимальный

    def build_zero_field_term(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        F = torch.zeros((*self.batch_shape, self.dim, self.dim), dtype=torch.complex64, device=self.device)
        for e_idx, n_idx, A in self.hyperfine_interactions:
            A = A.tensor
            F += scalar_tensor_multiplication(self._operator_cache[e_idx],
                                              self._operator_cache[len(self.electrons) + n_idx], A)
        F = F
        return F

    def _build_electron_zeeman_terms(self) -> torch.Tensor:
        """Constructs the Zeeman interaction terms Gx, Gy, Gz. for electron spins with give g-tensors"""
        G = torch.zeros((*self.batch_shape, 3, self.dim, self.dim), dtype=torch.complex64, device=self.device)
        for idx, g_tensor in enumerate(self.g_tensors):
            g = g_tensor.tensor
            G += transform_tensor_components(self._operator_cache[idx], g)
        G *= (constants.BOHR / constants.PLANCK)
        return G

    def _build_nucleus_zeeman_terms(self) -> torch.Tensor:
        """Constructs the Nucleus interaction terms Gx, Gy, Gz. for nucleus spins"""
        G = torch.zeros((*self.batch_shape, 3, self.dim, self.dim), dtype=torch.complex64, device=self.device)
        for idx, nucleus in enumerate(self.nuclei):
            g = nucleus.g_factor
            G += self._operator_cache[len(self.electrons) + idx] * g
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
        for idx, electron in enumerate(self.electrons):
            electron_contrib += electron.spin * torch.sum(
                self.g_tensors[idx].tensor[..., :, 0], dim=-1, keepdim=True).abs()

        nuclei_contrib = 0
        for idx, nucleus in enumerate(self.nuclei):
            nuclei_contrib += nucleus.spin * nucleus.g_factor.abs()

        return (electron_contrib * (constants.BOHR / constants.PLANCK) +
            nuclei_contrib * (constants.NUCLEAR_MAGNETRON / constants.PLANCK)).unsqueeze(-1)


    # Потом нужно переделать. Есть небольшой оверхэд
    def build_field_dep_stained(self):
        """
        Calculate electron Zeeman field dependant strained part
        :return:
        """
        for idx, g_tensor in enumerate(self.g_tensors):
            g = g_tensor.strained_tensor
            if g is None:
                pass
            else:
                for part in range(3):
                    # yield transform_tensor_components(self._operator_cache[idx], g[..., part, :, :])[..., 2, :, :] *\
                    #    (constants.BOHR / constants.PLANCK)
                    yield torch.einsum("...j, jkl->...kl", g[..., part, 2, :], self._operator_cache[idx]) *\
                        (constants.BOHR / constants.PLANCK)

    def build_zero_field_stained(self) -> torch.Tensor:
        """Constructs the zero-field strained part."""
        for e_idx, n_idx, hyperfine in self.hyperfine_interactions:
            hyperfine_stained = hyperfine.strained_tensor
            if hyperfine_stained is None:
                pass
            else:
                for part in range(3):
                    yield scalar_tensor_multiplication(
                        self._operator_cache[e_idx],
                        self._operator_cache[len(self.electrons) + n_idx],
                        hyperfine_stained[..., part, :, :])

    def get_hamiltonian_terms(self) -> tuple:
        """
        Returns F, Gx, Gy, Gz.
        F is magnetic field free term
        Gx, Gy, Gz are terms multiplied to Bx, By, Bz respectively

        """
        return self.build_zero_field_term(), *self.build_zeeman_terms()



class MultiOrientedSystem():
    def __init__(self, spin_system: SpinSystem, humiltonian_strained: torch.Tensor, rotation_matrices: torch.Tensor):
        self.humiltonian_strained =\
            (self._expand_hamiltonian(humiltonian_strained,
                                      self.orientation_vector(rotation_matrices)
                                      ) ** 2).sum(dim=-1).square()
        self.spin_system = spin_system
        self.replace_to_oriented_interactions(rotation_matrices)


    def _expand_hamiltonian(self, humiltonian_strained, orientation_vector):
        return torch.einsum("...i, ri -> ...ri", humiltonian_strained, orientation_vector)


    def replace_to_oriented_interactions(self, rotation_matrices):
        rotation_matrices = rotation_matrices.to(torch.complex64)
        batche_shape = torch.Size([*self.spin_system.batch_shape, rotation_matrices.shape[0]])
        self.spin_system.batch_shape = batche_shape

        g_tensors = [g_tensor.tensor for g_tensor in self.spin_system.g_tensors]
        g_strained = [
            g_tensor.strained_tensor for g_tensor in self.spin_system.g_tensors if g_tensor.strained_tensor is not None]
        hyperfine_tensors = [interaction[-1].tensor for interaction in self.spin_system.hyperfine_interactions]
        hyperfine_strained = [
            interaction[-1].strained_tensor for interaction in self.spin_system.hyperfine_interactions
            if interaction[-1].strained_tensor is not None]
        interactions = torch.stack((*g_tensors, *hyperfine_tensors), dim=0)
        interactions = utils.apply_expanded_rotations(rotation_matrices, interactions)
        g_tensors, hyperfine_interactions =\
            torch.split(interactions, [len(g_tensors), len(hyperfine_tensors)], dim=0)

        to_stain = (*g_strained, *hyperfine_strained)
        if to_stain:
            strained = torch.stack(to_stain, dim=0)
            strained = utils.apply_expanded_rotations(rotation_matrices, strained)
            strained = strained.transpose(-3, -4)
            g_strained, hyperfine_strained =\
                torch.split(strained, [len(g_strained), len(hyperfine_strained)], dim=0)
        else:
            g_strained = [None]
            hyperfine_strained = [None]

        for idx in range(len(self.spin_system.g_tensors)):
            strained_idx = 0
            if self.spin_system.g_tensors[idx].strained_tensor is not None:
                self.spin_system.g_tensors[idx] =\
                    MultiOrientedInteraction(g_tensors[idx], g_strained[strained_idx], batche_shape)
                strained_idx += 1
            else:
                self.spin_system.g_tensors[idx] =\
                    MultiOrientedInteraction(g_tensors[idx], None, batche_shape)


        for idx in range(len(self.spin_system.hyperfine_interactions)):
            strained_idx = 0
            el, nucs, interaction = self.spin_system.hyperfine_interactions[idx]
            if interaction.strained_tensor is not None:
                self.spin_system.hyperfine_interactions[idx] = \
                    (el, nucs, MultiOrientedInteraction(hyperfine_interactions[idx],
                                                        hyperfine_strained[strained_idx], batche_shape))
                strained_idx += 1
            else:
                self.spin_system.hyperfine_interactions[idx] = \
                    (el, nucs, MultiOrientedInteraction(hyperfine_interactions[idx], None, batche_shape))
                strained_idx += 1

    def get_hamiltonian_terms(self) -> tuple:
        """
        Returns F, Gx, Gy, Gz.
        F is magnetic field free term
        Gx, Gy, Gz are terms multiplied to Bx, By, Bz respectively

        """
        return self.spin_system.get_hamiltonian_terms()


    def calculate_derivative_max(self) -> tuple:
        """
        Calculate the maximum value of the energy derivatives with respect to magnetic field.
        It is assumed that B has direction along z-axis
        :return: the maximum value of the energy derivatives with respect to magnetic field
        """
        return self.spin_system.calculate_derivative_max()


    def build_field_dep_stained(self):
        """
        Calculate electron Zeeman field dependant strained part
        :return:
        """
        return self.spin_system.build_field_dep_stained()


    def build_zero_field_stained(self) -> torch.Tensor:
        """Constructs the zero-field strained part."""
        return self.spin_system.build_zero_field_stained()

    def orientation_vector(self, rotation_matrices):
        return rotation_matrices[..., :, -1]

    def build_hamiltonian_stained(self) -> torch.Tensor:
        """Constructs the zero-field strained part."""

        return self.humiltonian_strained


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
    g_isotropic = torch.diag(torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)) + torch.linspace(0.01, 3, 1000)[..., None, None]
    nucleus = particles.Nucleus("15N")
    A = torch.eye(3, dtype=torch.float32) * 7 * 10**7

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
    B_high = torch.tensor([100.0]).expand(1000)
    B_low = torch.tensor([0.01]).expand(1000)
    system = system_2()
    F, Gx, Gy, Gz = system.get_hamiltonian_terms()
    deriv_max = system.calculate_derivative_max()[..., None]
    resonance_interval_finder = res_field.GeneralResonanceIntervalSolver()
    baselign_sign = res_field.compute_zero_field_resonance(F, resonance_frequency)
    res_loc = res_field.ResonanceLocator()
    batches = resonance_interval_finder.get_resonance_intervals(F, Gz, B_low, B_high,
                                            resonance_frequency, baselign_sign, deriv_max)
    res_loc.locate_resonance_fields(batches, resonance_frequency)

