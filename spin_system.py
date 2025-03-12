import functools
import itertools
from abc import ABC, abstractmethod
import copy

import torch

import constants
import particles
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
        '...ij, jkl, ikl->...kl',
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
    def config_shape(self):
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
    def config_shape(self):
        return self.shape[:-1]


class MultiOrientedInteraction(BaseInteraction):
    def __init__(self, oriented_tensor, strained_tensor, config_shape):
        self._oriented_tensor = oriented_tensor
        self._strained_tensor = strained_tensor
        self._config_shape = config_shape

    @property
    def tensor(self):
        return self._oriented_tensor

    @property
    def strained_tensor(self):
        return self._strained_tensor

    @property
    def config_shape(self):
        return self.tensor.shape[:-2]


class SpinSystem:
    """Represents a spin system with electrons, nuclei, and interactions."""
    def __init__(self, electrons: list[particles.Electron], g_tensors: list[BaseInteraction],
                 electron_nuclei: list[tuple[int, int, BaseInteraction]] | None = None,
                 electron_electron: list[tuple[int, int, BaseInteraction]] | None = None,
                 nuclei_nuclei: list[tuple[int, int, BaseInteraction]] | None = None,
                 nuclei: list[particles.Nucleus] | None = None,
                 device=torch.device("cpu")):

        self.electrons = electrons
        self.g_tensors = g_tensors
        self.nuclei = nuclei if nuclei else []

        self.electron_nuclei = electron_nuclei if electron_nuclei else []
        self.electron_electron = electron_electron if electron_electron else []
        self.nuclei_nuclei = nuclei_nuclei if nuclei_nuclei else []

        if len(self.g_tensors) != len(self.electrons):
            raise ValueError("the number of g tensors must be equal to the number of electrons")

        self._operator_cache = {}  # Format: {particle_idx: tensor}
        self._precompute_all_operators()
        self.device = device

    @property
    def config_shape(self) -> int:
        """shape of the tensor"""
        return self.g_tensors[0].config_shape

    @property
    def dim(self) -> int:
        """Dimension of the system's Hilbert space."""
        return torch.prod(
            torch.tensor([int(2 * p.spin + 1) for p in itertools.chain(self.electrons, self.nuclei)])
        ).item()

    @property
    def operator_cache(self):
        return self._operator_cache

    #  Нужно передалать функцию, слишком много циклов. Можно улучшить
    def _precompute_all_operators(self):
        """Precompute spin operators for all particles in the full Hilbert space."""
        particels = self.electrons + self.nuclei
        for idx, p in enumerate(particels):
            axis_cache = []
            for axis, mat in zip(['x', 'y', 'z'], p.spin_matrices):
                operator = create_operator(particels, idx, mat)
                axis_cache.append(operator)
            self.operator_cache[idx] = torch.stack(axis_cache, dim=-3)   # Сейчас каждый спин даёт матрицу [K, K] и
                                                                         # расчёт взаимодействией не оптимальный

class BaseSample():
    def __init__(self, spin_system: SpinSystem, *args, **kwargs):
        self.spin_system = copy.deepcopy(spin_system)

    @property
    def config_shape(self):
        return self.spin_system.config_shape

    def build_electron_electron(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        F = torch.zeros((*self.config_shape, self.spin_system.dim, self.spin_system.dim), dtype=torch.complex64,
                        device=self.spin_system.device)
        for e_idx_1, e_idx_2, interaction in self.spin_system.electron_electron:
            interaction = interaction.tensor
            F += scalar_tensor_multiplication(
                self.spin_system.operator_cache[e_idx_1],
                self.spin_system.operator_cache[e_idx_2],
                interaction)
        return F

    def build_electron_nuclei(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        F = torch.zeros((*self.config_shape, self.spin_system.dim, self.spin_system.dim), dtype=torch.complex64,
                        device=self.spin_system.device)
        for e_idx, n_idx, interaction in self.spin_system.electron_nuclei:
            interaction = interaction.tensor
            F += scalar_tensor_multiplication(
                self.spin_system.operator_cache[e_idx],
                self.spin_system.operator_cache[len(self.spin_system.electrons) + n_idx],
                interaction)
        return F

    def build_nuclei_nuclei(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        F = torch.zeros((*self.config_shape, self.spin_system.dim, self.spin_system.dim), dtype=torch.complex64,
                        device=self.spin_system.device)
        for n_idx_1, n_idx_2, interaction in self.spin_system.electron_nuclei:
            interaction = interaction.tensor
            F += scalar_tensor_multiplication(
                self.spin_system.operator_cache[len(self.spin_system.electrons) + n_idx_1],
                self.spin_system.operator_cache[len(self.spin_system.electrons) + n_idx_2],
                interaction)
        return F

    def build_first_order_interactions(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F of the first order operators"""
        return self.build_nuclei_nuclei() + self.build_electron_nuclei() + self.build_electron_electron()

    def build_zero_field_term(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        return self.build_first_order_interactions()

    def _build_electron_zeeman_terms(self) -> torch.Tensor:
        """Constructs the Zeeman interaction terms Gx, Gy, Gz. for electron spins with give g-tensors"""
        G = torch.zeros((*self.config_shape, 3, self.spin_system.dim, self.spin_system.dim), dtype=torch.complex64,
                        device=self.spin_system.device)
        for idx, g_tensor in enumerate(self.spin_system.g_tensors):
            g = g_tensor.tensor
            G += transform_tensor_components(self.spin_system.operator_cache[idx], g)
        G *= (constants.BOHR / constants.PLANCK)
        return G

    def _build_nucleus_zeeman_terms(self) -> torch.Tensor:
        """Constructs the Nucleus interaction terms Gx, Gy, Gz. for nucleus spins"""
        G = torch.zeros((*self.config_shape, 3, self.spin_system.dim, self.spin_system.dim), dtype=torch.complex64,
                        device=self.spin_system.device)
        for idx, nucleus in enumerate(self.spin_system.nuclei):
            g = nucleus.g_factor
            G += self.spin_system.operator_cache[len(self.spin_system.electrons) + idx] * g
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
        for idx, electron in enumerate(self.spin_system.electrons):
            electron_contrib += electron.spin * torch.sum(
                self.spin_system.g_tensors[idx].tensor[..., :, 0], dim=-1, keepdim=True).abs()

        nuclei_contrib = 0
        for idx, nucleus in enumerate(self.spin_system.nuclei):
            nuclei_contrib += nucleus.spin * nucleus.g_factor.abs()

        return (electron_contrib * (constants.BOHR / constants.PLANCK) +
            nuclei_contrib * (constants.NUCLEAR_MAGNETRON / constants.PLANCK)).unsqueeze(-1)

    def get_hamiltonian_terms(self) -> tuple:
        """
        Returns F, Gx, Gy, Gz.
        F is magnetic field free term
        Gx, Gy, Gz are terms multiplied to Bx, By, Bz respectively

        """
        return self.build_zero_field_term(), *self.build_zeeman_terms()


    def build_field_dep_staine(self):
        """
        Calculate electron Zeeman field dependant strained part
        :return:
        """
        for idx, g_tensor in enumerate(self.spin_system.g_tensors):
            g = g_tensor.strained_tensor
            if g is None:
                pass
            else:
                yield (
                    self.spin_system.operator_cache[idx],
                    g[..., :, 2, :])

    def build_zero_field_staine(self) -> torch.Tensor:
        """Constructs the zero-field strained part."""
        for e_idx, n_idx, electron_nuclei in self.spin_system.electron_nuclei:
            electron_nuclei = electron_nuclei.strained_tensor
            if electron_nuclei is None:
                pass
            else:
                yield (
                    self.spin_system.operator_cache[e_idx],
                    self.spin_system.operator_cache[len(self.spin_system.electrons) + n_idx],
                    electron_nuclei)



class SpinSystemOrientator:
    def __call__(self, spin_system: SpinSystem, rotation_matrices: torch.Tensor) -> SpinSystem:
        """
        :param spin_system: spin_system with interactions
        :param rotation_matrices: rotation_matrices that rotate spin system
        :return: modified spin system with all rotated interactions
        """
        spin_system = self.transform_spin_system_to_oriented(copy.deepcopy(spin_system), rotation_matrices)
        return spin_system

    def interactions_to_multioriented(self, interactions: list[BaseInteraction], rotation_matrices: torch.Tensor):
        interactions_tensors = torch.stack([interaction.tensor for interaction in interactions], dim=0)
        interactions_tensors = utils.apply_expanded_rotations(rotation_matrices, interactions_tensors)
        not_none_strained = [
            interaction.strained_tensor for interaction in interactions if interaction.strained_tensor is not None
        ]
        none_strained_flag = [
            True if interaction.strained_tensor is None else False for interaction in interactions
        ]
        if not_none_strained:
            strained_tensors = torch.stack(not_none_strained, dim=0)
            strained_tensors = utils.apply_expanded_rotations(rotation_matrices, strained_tensors)
            strained_tensors = strained_tensors.transpose(-3, -4)
            strained_iterator = iter(strained_tensors)
            strined_res = [None if x else next(strained_iterator) for x in none_strained_flag]
        else:
            strined_res = [None] * len(interactions)
        return interactions_tensors, strined_res


    def _apply_reverse_transform(self, spin_system: SpinSystem, new_interactions: list):
        # Determine how many interactions belong to each original group.
        num_g = len(spin_system.g_tensors)
        num_nuc = len(spin_system.electron_nuclei)
        num_el = len(spin_system.electron_electron)

        # Split the new interactions list into the three groups.
        new_g_tensors = new_interactions[:num_g]
        new_electron_nuclei = new_interactions[num_g:num_g + num_nuc]
        new_electron_electron = new_interactions[num_g + num_nuc:]

        spin_system.g_tensors = [
            interaction for interaction in new_g_tensors
        ]
        spin_system.electron_nuclei = [
            (x, y, interaction) for (x, y, _), interaction in zip(spin_system.electron_nuclei, new_electron_nuclei)
        ]
        spin_system.electron_electron = [
            (x, y, interaction) for (x, y, _), interaction in zip(spin_system.electron_electron, new_electron_electron)
        ]
        return spin_system

    def transform_spin_system_to_oriented(self, spin_system: SpinSystem, rotation_matrices: torch.Tensor):
        rotation_matrices = rotation_matrices.to(torch.complex64)
        config_shape = torch.Size([*spin_system.config_shape, rotation_matrices.shape[0]])
        interactions = [g_tensor for g_tensor in spin_system.g_tensors] + \
                       [el_nuc[-1] for el_nuc in spin_system.electron_nuclei] +\
                       [el_el[-1] for el_el in spin_system.electron_electron]
        interactions_tensors, strained_tensors = self.interactions_to_multioriented(interactions, rotation_matrices)
        interactions =\
            [MultiOrientedInteraction(interactions_tensor, strained_tensor, config_shape) for
             interactions_tensor, strained_tensor in zip(interactions_tensors, strained_tensors)]
        spin_system = self._apply_reverse_transform(spin_system, interactions)
        return spin_system





class MultiOrientedSample(BaseSample):
    def __init__(self, spin_system: SpinSystem, humiltonian_strained: torch.Tensor, rotation_matrices: torch.Tensor):
        super().__init__(spin_system)
        self.humiltonian_strained = self._expand_hamiltonian(
            humiltonian_strained,
            self.orientation_vector(rotation_matrices)
        )
        self.spin_system = SpinSystemOrientator()(spin_system, rotation_matrices)



    def _expand_hamiltonian(self, humiltonian_strained, orientation_vector):
        return torch.einsum("...i, ri -> ...r", humiltonian_strained**2, orientation_vector**2).sqrt()

    def orientation_vector(self, rotation_matrices):
        return rotation_matrices[..., :, -1]


    def build_hamiltonian_stained(self) -> torch.Tensor:
        """Constructs the zero-field strained part."""
        return self.humiltonian_strained

