import functools
import itertools
from abc import ABC, abstractmethod
import copy
import typing as tp
import pickle

import json
import numpy as np
from pathlib import Path
import yaml

import torch
import scipy

import constants
import mesher
import particles
import utils

from mesher import BaseMesh


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
        '...ij, jnl, ilm->...nm',
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
def init_tensor(
        components: tp.Union[torch.Tensor, tp.Sequence[float], float],  device: torch.device, dtype: torch.dtype
):
    if isinstance(components, torch.Tensor):
        tensor = components.to(device=device, dtype=dtype)
        if tensor.shape[-1] == 3:
            return tensor
        elif tensor.shape[-1] == 2:
            axis_val, z_val = tensor[0], tensor[1]
            return torch.tensor([axis_val, axis_val, z_val], device=device, dtype=dtype)
        elif tensor.shape[-1] == 1:
            value = tensor.item()
            return torch.full((3,), value, device=device, dtype=dtype)
        else:
            raise ValueError(f"Tensor must have shape [..., 3] or [1] or [2], got {tensor.shape}")

    elif isinstance(components, (list, tuple)):
        if len(components) == 1:
            value = components[0]
            return torch.full((3,), value, device=device, dtype=dtype)
        elif len(components) == 2:
            axis_val, z_val = components
            return torch.tensor([axis_val, axis_val, z_val], device=device, dtype=dtype)
        elif len(components) == 3:
            return torch.tensor(components, device=device, dtype=dtype)
        else:
            raise ValueError(f"List must have 1, 2, or 3 elements, got {len(components)}")

    elif isinstance(components, (int, float)):
        return torch.full((3,), components, device=device, dtype=dtype)

    else:
        raise TypeError(f"components must be a tensor, list, tuple, or scalar, got {type(components)}")


def init_de_tensor(
        components: tp.Union[torch.Tensor, tp.Sequence[float], float],  device: torch.device, dtype: torch.dtype
):
    if isinstance(components, torch.Tensor):
        tensor = components.to(device=device, dtype=dtype)
        if tensor.shape[-1] == 3:
            return tensor
        elif tensor.numel() == 2:
            D, E = tensor[0], tensor[1]
            Dx = - D / 3 + E
            Dy = - D / 3 - E
            Dz = 2 * D / 3
            return torch.tensor([Dx, Dy, Dz], device=device, dtype=dtype)
        elif tensor.numel() == 1:
            Dx = - tensor / 3
            Dy = - tensor / 3
            Dz = 2 * tensor / 3
            return torch.tensor([Dx, Dy, Dz], device=device, dtype=dtype)
        else:
            raise ValueError(f"Tensor must have shape [..., 3] or [1] or [2], got {tensor.shape}")

    elif isinstance(components, (list, tuple)):
        if len(components) == 1:
            value = components[0]
            Dx = - value / 3
            Dy = - value / 3
            Dz = 2 * value / 3
            return torch.tensor([Dx, Dy, Dz], device=device, dtype=dtype)

        elif len(components) == 2:
            D, E = components[0], components[1]
            Dx = - D / 3 + E
            Dy = - D / 3 - E
            Dz = 2 * D / 3
            return torch.tensor([Dx, Dy, Dz], device=device, dtype=dtype)

        elif len(components) == 3:
            return torch.tensor(components, device=device, dtype=dtype)
        else:
            raise ValueError(f"List must have 1, 2, or 3 elements, got {len(components)}")

    elif isinstance(components, (int, float)):
        return torch.full((3,), components, device=device, dtype=dtype)

    else:
        raise TypeError(f"components must be a tensor, list, tuple, or scalar, got {type(components)}")


class BaseInteraction(ABC):
    @property
    @abstractmethod
    def tensor(self):
        pass

    @property
    def components(self):
        return self.tensor

    @property
    def strain(self):
        return self.strained_tensor

    @property
    def frame(self):
        return None

    @property
    @abstractmethod
    def strained_tensor(self):
        pass

    @property
    @abstractmethod
    def config_shape(self):
        pass

    @property
    @abstractmethod
    def strain_correlation(self):
        """
        :return: The correlation matrix of strain parameters
        """
        pass

    def __len__(self):
        return len(self.components)

    def __repr__(self):
        if hasattr(self.components, 'tolist'):
            components_str = [f"{val:.2e}" if abs(val) >= 1e4 else f"{val:.4f}"
                              for val in self.components.tolist()]
        else:
            components_str = [f"{val:.2e}" if abs(val) >= 1e4 else f"{val:.4f}"
                              for val in self.components]

        lines = [
            f"Principal values: [{', '.join(components_str)}]",
        ]

        if self.frame is not None:
            if hasattr(self.frame, 'tolist'):
                frame_vals = self.frame.tolist()
            else:
                frame_vals = self.frame

            if len(frame_vals) == 3:  # Euler angles
                frame_str = f"[α={frame_vals[0]:.3f}, β={frame_vals[1]:.3f}, γ={frame_vals[2]:.3f}] rad"
            else:
                frame_str = str(frame_vals)
            lines.append(f"Frame (Euler angles): {frame_str}")
        else:
            lines.append("Frame: Identity (no rotation)")

        if self.strain is not None:
            if hasattr(self.strain, 'tolist'):
                strain_vals = self.strain.tolist()
                strain_str = [f"{val:.2e}" if abs(val) >= 1e4 else f"{val:.4f}"
                              for val in strain_vals]
                lines.append(f"Strain: [{', '.join(strain_str)}]")
            else:
                lines.append(f"Strain: {self.strain}")
        else:
            lines.append("Strain: None")

        return '\n'.join(lines)


class Interaction(BaseInteraction):
    def __init__(self, components: tp.Union[torch.Tensor, tp.Sequence, float],
                 frame: tp.Optional[tp.Union[torch.Tensor, tp.Sequence]] = None,
                 strain: tp.Optional[tp.Union[torch.Tensor, tp.Sequence, float]] = None,
                 device=torch.device("cpu"), dtype=torch.float32):
        """
        :param components:
        torch.Tensor | Sequence[float] | float
            The tensor components, provided in one of the following forms:
              - A scalar (for isotropic interaction).
              - A sequence of two values (axial and z components).
              - A sequence of three values (principal components).
        The possible units are [T, Hz, dimensionless]

        :param frame:
        torch.Tensor | Sequence[float] optional
            Orientation of the tensor. Can be provided as:
              - A 1D tensor of shape (3,) representing Euler angles in ZYZ' convention.
              - A 2D tensor of shape (3, 3) representing a rotation matrix.
            Default is `None`, meaning lab frame.

        :param strain:
        torch.Tensor| Sequence[float] | float, optional
            Parameters describing interaction broadening or distribution.
            Default is `None`.

        :param device:

        :param dtype:
        """

        self._components = init_tensor(components, device=device, dtype=dtype)
        self.shape = self._components.shape
        batch_shape = self._components.shape[:-1]

        self._construct_rot_matrix(frame, batch_shape)

        self._strain = init_tensor(strain, device=device, dtype=dtype) if strain is not None else None

        if (self._strain is not None) and (self._strain.shape != self.shape):
            raise ValueError("The strain shape must be equal to shape of the initial components."
                             "Please point it as x, y, z components")

        self._strain_correlation = torch.eye(3, device=device, dtype=dtype)

    def _construct_rot_matrix(self, frame: tp.Optional[torch.tensor], batch_shape):
        if frame is None:
            self._frame = torch.zeros((*batch_shape, 3), dtype=torch.float32)  # alpha, beta, gamma
            self._rot_matrix = self.euler_to_rotmat(self._frame).to(self.components.dtype)

        else:
            if not isinstance(frame, torch.Tensor):
                raise TypeError("frame must be a torch.Tensor or None.")

            if frame.shape[-2:] == (3, 3):
                if frame.shape[:-2] != batch_shape:
                    raise ValueError(
                        f"rotation‐matrix batch dims {frame.shape[:-2]} "
                        f"must match components batch dims {batch_shape}"
                    )
                self._frame = utils.rotation_matrix_to_euler_angles(frame)
                self._rot_matrix = frame.to(self.components.dtype)

            elif frame.shape == (*batch_shape, 3):
                self._frame = frame.to(torch.float32)
                self._rot_matrix = self.euler_to_rotmat(self._frame).to(self.components.dtype)

            else:
                raise ValueError(
                    "frame must be either:\n"
                    "  • None (→ identity rotation),\n"
                    "  • a tensor of Euler angles with shape batch×3,\n"
                    "  • or a tensor of rotation matrices with shape batch×3×3."
                )

    def euler_to_rotmat(self, euler_angles: torch.Tensor):
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

    def _strained_tensor(self) -> tp.Optional[torch.Tensor]:
        """
        :return: return the None or the tensor with the shape [..., 3, 3, 3] or None
        """
        if self._strain is None:
            return None
        else:
            return self._strain.unsqueeze(-1).unsqueeze(-1) *\
               torch.einsum("...ik, ...jk->...kij", self._rot_matrix, self._rot_matrix)


    @property
    def tensor(self):
        """
        :return: the full tensor of interaction with shape [..., 3, 3] with applied rotation
        """
        return self._tensor()

    @property
    def strain(self):
        """
        :return: None or tensor with shape [..., 3]
        """
        return self._strain

    @property
    def strained_tensor(self):
        """
        :return: None or the tensor with shape [...., 3, 3, 3], where first '3' is x, y, z components
        """
        return self._strained_tensor()

    @property
    def config_shape(self):
        return self.shape[:-1]

    @property
    def components(self):
        """
        :return: tensor with shape [..., 3] - the principle components of a tensor
        """
        return self._components

    @property
    def frame(self):
        """
        :return: angles with ZYZ' notation.
        """
        return self._frame

    @frame.setter
    def frame(self, frame):
        if frame is None:
            self._frame = torch.tensor([0.0, 0.0, 0.0])  # alpha, beta, gamma
        self._rot_matrix = self.euler_to_rotmat(self._frame)

    @property
    def strain_correlation(self):
        """
        In some cases the components of the interaction can correlate.
        To implement this correlation the strain_correlation matrix is used. For example, in the case of D/E interaction
        strain_correlation = [[-1/3, 1], [-1/3, -1], [2/3, 0]] - the matrix of trnasformation of Dx, Dy, Dz to D and E
        :return:
        """
        return self._strain_correlation

    @strain_correlation.setter
    def strain_correlation(self, correlation_matrix: torch.Tensor):
        self._strain_correlation = correlation_matrix

    def __add__(self, other):
        if not isinstance(other, Interaction):
            raise TypeError("Can only add Interaction objects together")

        if torch.allclose(self.frame, other.frame, atol=1e-6):
            new_frame = self.frame.clone()
            new_components = self.components + other.components
        else:
            tensor_self = self._tensor()
            tensor_other = other._tensor()  # [..., 3, 3]
            combined_tensor = tensor_self + tensor_other

            eigenvalues, eigenvectors = torch.linalg.eigh(combined_tensor)

            sorted_indices = torch.argsort(eigenvalues, dim=-1, descending=True)
            new_components = torch.gather(eigenvalues, -1, sorted_indices)

            new_rot_matrix = eigenvectors[..., sorted_indices].transpose(-2, -1)
            new_frame = utils.rotation_matrix_to_euler_angles(new_rot_matrix)

        if self.strain is not None and other.strain is not None:
            new_strain = self.strain + other.strain
            correlation_matrix = torch.cat((self.strain_correlation, other.strain_correlation), dim=-1)
            #  It is suggested that D,E are not correlated

        elif self.strain is not None:
            new_strain = self.strain.clone()
            correlation_matrix = self.strain_correlation.clone()

        elif other.strain is not None:
            new_strain = other.strain.clone()
            correlation_matrix = other.strain_correlation.clone()

        else:
            new_strain = None
            correlation_matrix = torch.eye(3, dtype=new_components.dtype, device=new_components.device)

        interaction = Interaction(
            components=new_components,
            frame=new_frame,
            strain=new_strain,
            device=self.components.device,
            dtype=self.components.dtype
        )
        interaction.strain_correlation = correlation_matrix
        return interaction


class DEInteraction(Interaction):
    def __init__(self, components: torch.Tensor,
                 frame: torch.Tensor = None, strain: torch.Tensor = None,
                 device=torch.device("cpu"), dtype=torch.float32):
        """
        DEInteraction is given by two components D and E. To transform to x, y, z components the next equation is used:
        Dx = -D * 1/3 + E
        Dy = -D * 1/3 - E
        Dz = D * 2/3

                Note on DE Interaction vs. Simple Interaction
        The DE Interaction is equivalent to simple Interaction in terms of components when
        the trace of the tensor equals zero,
        but they are not equivalent in terms of strains.
        In DE Interaction, the D and E components (or only D) have a distribution,
        whereas in simple interaction, the components Dx, Dy, and Dz are distributed.

        :param components:
        torch.Tensor | Sequence[float] | float
            The tensor components, provided in one of the following forms:
              - A scalar. It is only D value.
              - A sequence of two values (D and E values).
        The possible units are [T, Hz, dimensionless]

        :param frame:
        torch.Tensor | Sequence[float] optional
            Orientation of the tensor. Can be provided as:
              - A 1D tensor of shape (3,) representing Euler angles in ZYZ' convention.
              - A 2D tensor of shape (3, 3) representing a rotation matrix.
            Default is `None`, meaning lab frame.

        :param strain:
        torch.Tensor| Sequence[float] | float, optional
            Parameters describing interaction broadening or distribution.
            Default is `None`.

        :param device:

        :param dtype:
        """
        components = init_de_tensor(components, device, dtype)
        strain = init_de_tensor(strain, device=device, dtype=dtype) if strain is not None else None
        super().__init__(components, frame, strain, device, dtype)

        # D, E to Dx, Dy, Dz transformation
        self._strain_correlation = torch.tensor([[-1/3, 1], [-1/3, -1], [2/3, 0]], dtype=dtype, device=device)



class MultiOrientedInteraction(BaseInteraction):
    def __init__(self, oriented_tensor, strained_tensor, config_shape, strain_correlation: torch.Tensor):
        self._oriented_tensor = oriented_tensor
        self._strained_tensor = strained_tensor
        self._config_shape = config_shape
        self._strain_correlation = strain_correlation

    @property
    def tensor(self):
        return self._oriented_tensor

    @property
    def strained_tensor(self):
        return self._strained_tensor

    @property
    def config_shape(self):
        return self.tensor.shape[:-2]

    @property
    def strain_correlation(self):
        return self._strain_correlation

    @strain_correlation.setter
    def strain_correlation(self, correlation_matrix: torch.Tensor):
        self._strain_correlation = correlation_matrix


class SpinSystem:
    """Represents a spin system with electrons, nuclei, and interactions."""
    def __init__(self, electrons: tp.Union[list[particles.Electron], list[float]],
                 g_tensors: list[BaseInteraction],
                 nuclei: tp.Optional[tp.Union[list[particles.Nucleus], list[str]]] = None,
                 electron_nuclei: list[tuple[int, int, BaseInteraction]] | None = None,
                 electron_electron: list[tuple[int, int, BaseInteraction]] | None = None,
                 nuclei_nuclei: list[tuple[int, int, BaseInteraction]] | None = None,
                 device=torch.device("cpu")):

        """
        :param electrons:
        list[Electron] | list[float]
            Electron spins in the system. Can be specified as:
              - A list of `Electron` particle instances.
              - A list of spin quantum numbers (e.g., [0.5, 1.0]).

        :param g_tensors:
        list[BaseInteraction]
            g-tensors corresponding to each electron in `electrons`.
            Each element must be an instance of `BaseInteraction` (e.g., `Interaction`).

        :param nuclei:
        list[Nucleus] | list[str], optional
            Nuclei in the system. Can be given as:
              - A list of `Nucleus` particle instances.
              - A list of isotope symbols (e.g., ["1H", "13C"]).
            Default is `None` (no nuclei).

        :param electron_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Hyperfine interactions between electrons and nuclei.
            Each tuple is of the form (electron_index, nucleus_index, interaction_tensor).
            Default is `None`.

        :param electron_electron:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or exchange interactions between pairs of electrons.
            Each tuple is of the form (electron_index, electron_index, interaction_tensor).
            Default is `None`.

        :param nuclei_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or J-coupling interactions between pairs of nuclei.
            Each tuple is of the form (nucleus_index, nucleus_index, interaction_tensor).
            Default is `None`

        :param device:
        """

        self.electrons = self._init_electrons(electrons)
        self.g_tensors = g_tensors
        self.nuclei = self._init_nuclei(nuclei) if nuclei else []

        self.electron_nuclei = electron_nuclei if electron_nuclei else []
        self.electron_electron = electron_electron if electron_electron else []
        self.nuclei_nuclei = nuclei_nuclei if nuclei_nuclei else []

        if len(self.g_tensors) != len(self.electrons):
            raise ValueError("the number of g tensors must be equal to the number of electrons")

        self._operator_cache = {}  # Format: {particle_idx: tensor}
        self._precompute_all_operators()
        self.device = device

    def _init_electrons(self, electrons):
        return [particles.Electron(electron) if isinstance(electron, float) else electron for electron in electrons]

    def _init_nuclei(self, nuclei: tp.Union[list[particles.Nucleus], list[str]]):
        return [particles.Nucleus(nucleus) if isinstance(nucleus, str) else nucleus for nucleus in nuclei]

    @property
    def config_shape(self) -> tp.Iterable:
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

    def update(self,
               g_tensors: list[BaseInteraction] = None,
               electron_nuclei: list[tuple[int, int, BaseInteraction]] | None = None,
               electron_electron: list[tuple[int, int, BaseInteraction]] | None = None,
               nuclei_nuclei: list[tuple[int, int, BaseInteraction]] | None = None):
        """
        Update the parameters of spin system. No recomputation of spin vectors does not occur
        :param g_tensors:
        list[BaseInteraction]
            g-tensors corresponding to each electron in `electrons`.
            Each element must be an instance of `BaseInteraction` (e.g., `Interaction`).

        :param electron_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Hyperfine interactions between electrons and nuclei.
            Each tuple is of the form (electron_index, nucleus_index, interaction_tensor).
            Default is `None`.

        :param electron_electron:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or exchange interactions between pairs of electrons.
            Each tuple is of the form (electron_index, electron_index, interaction_tensor).
            Default is `None`.

        :param nuclei_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or J-coupling interactions between pairs of nuclei.
            Each tuple is of the form (nucleus_index, nucleus_index, interaction_tensor).
            Default is `None`
        """

        self.g_tensors = g_tensors if g_tensors is not None else self.g_tensors
        self.electron_nuclei = electron_nuclei if electron_nuclei is not None else []
        self.electron_electron = electron_electron if electron_electron is not None else []
        self.nuclei_nuclei = nuclei_nuclei if nuclei_nuclei is not None else []

    def __repr__(self):
        hz_to_mhz = 1e-6
        T_to_mt = 1e3

        lines = ["=" * 60]
        lines.append("SPIN SYSTEM SUMMARY")
        lines.append("=" * 60)

        lines.append("\nPARTICLES:")
        lines.append("-" * 20)

        if self.electrons:
            electron_info = []
            for i, electron in enumerate(self.electrons):
                spin_str = f"S={electron.spin} \n"
                g_gactor_str = str(self.g_tensors[i]).replace('\n', '\n      ')
                spin_str += g_gactor_str
                electron_info.append(f"  e{i}: {spin_str}")

            lines.append(f"Electrons ({len(self.electrons)}):")
            lines.extend(electron_info)
        else:
            lines.append("Electrons: None")

        if self.nuclei:
            lines.append(f"\nNuclei ({len(self.nuclei)}):")
            for i, nucleus in enumerate(self.nuclei):
                nucleus_info = f"  n{i}: "
                if hasattr(nucleus, 'isotope'):
                    nucleus_info += f"{nucleus.isotope}, "
                if hasattr(nucleus, 'spin'):
                    nucleus_info += f"I={nucleus.spin}"
                lines.append(nucleus_info)
        else:
            lines.append("\nNuclei: None")

        lines.append(f"\nSYSTEM PROPERTIES:")
        lines.append("-" * 20)
        lines.append(f"Hilbert space dimension: {self.dim}")
        lines.append(f"Configuration shape: {tuple(self.config_shape)}")

        # Interactions section
        total_interactions = (len(self.electron_nuclei) +
                              len(self.electron_electron) +
                              len(self.nuclei_nuclei))

        if total_interactions > 0:
            lines.append(f"\nINTERACTIONS ({total_interactions} total):")
            lines.append("-" * 30)

            # Electron-nuclei interactions
            if self.electron_nuclei:
                lines.append(f"\nElectron-Nucleus ({len(self.electron_nuclei)}):")
                for i, (e_idx, n_idx, interaction) in enumerate(self.electron_nuclei):
                    lines.append(f"  {i + 1}. e{e_idx} ↔ n{n_idx}:")
                    interaction_str = str(interaction).replace('\n', '\n      ')
                    lines.append(f"      {interaction_str}")

            # Electron-electron interactions
            if self.electron_electron:
                lines.append(f"\nElectron-Electron ({len(self.electron_electron)}):")
                for i, (e1_idx, e2_idx, interaction) in enumerate(self.electron_electron):
                    lines.append(f"  {i + 1}. e{e1_idx} ↔ e{e2_idx}:")
                    interaction_str = str(interaction).replace('\n', '\n      ')
                    lines.append(f"      {interaction_str}")

            # Nucleus-nucleus interactions
            if self.nuclei_nuclei:
                lines.append(f"\nNucleus-Nucleus ({len(self.nuclei_nuclei)}):")
                for i, (n1_idx, n2_idx, interaction) in enumerate(self.nuclei_nuclei):
                    lines.append(f"  {i + 1}. n{n1_idx} ↔ n{n2_idx}:")
                    interaction_str = str(interaction).replace('\n', '\n      ')
                    lines.append(f"      {interaction_str}")
        else:
            lines.append(f"\nINTERACTIONS: None")

        lines.append("\n" + "=" * 60)
        return '\n'.join(lines)


class BaseSample:
    def __init__(self, spin_system: SpinSystem,
                 ham_strain: tp.Optional[tp.Union[torch.Tensor, float]] = None,
                 gauss: tp.Optional[tp.Union[torch.Tensor, float]] = None,
                 lorentz: tp.Optional[tp.Union[torch.Tensor, float]] = None, *args, **kwargs):
        """
        :param spin_system:
        SpinSystem
            The spin system describing electrons, nuclei, and their interactions.

        :param ham_strain:
        torch.Tensor, optional
            Anisotropic line width, due to the unresolved hyperfine interactions.
            The tensor components, provided in one of the following forms:
              - A scalar (for isotropic interaction).
              - A sequence of two values (axial and z components).
              - A sequence of three values (principal components).

        :param gauss:
        torch.Tensor, optional
            Gaussian broadening parameter(s). Defines inhomogeneous linewidth
            contributions (e.g., due to static disorder). Default is `None`.

        :param lorentz:
        torch.Tensor, optional
            Lorentzian broadening parameter(s). Defines homogeneous linewidth
            contributions (e.g., due to relaxation). Default is `None`

        :param args:
        :param kwargs:
        """

        self.base_spin_system = spin_system
        self.modified_spin_system = copy.deepcopy(spin_system)
        self._ham_strain = self._init_ham_str(ham_strain)
        self.base_ham_strain = copy.deepcopy(self._ham_strain)

        if gauss is None:
            self.gauss = torch.tensor(0.0, device=spin_system.device)
        else:
            self.gauss = torch.tensor(gauss)

        if lorentz is None:
            self.lorentz = torch.tensor(0.0, device=spin_system.device)
        else:
            self.lorentz = torch.tensor(lorentz)

    def _init_ham_str(self, ham_strain: torch.Tensor):
        if ham_strain is None:
            ham_strain = torch.zeros(3, dtype=torch.float32)
        else:
            ham_strain = init_tensor(ham_strain, device=torch.device("cpu"), dtype=torch.float32)
        return ham_strain

    def update(self,
               g_tensors: list[BaseInteraction] = None,
               electron_nuclei: list[tuple[int, int, BaseInteraction]] | None = None,
               electron_electron: list[tuple[int, int, BaseInteraction]] | None = None,
               nuclei_nuclei: list[tuple[int, int, BaseInteraction]] | None = None,
               ham_strain: tp.Optional[torch.Tensor] = None,
               gauss: tp.Union[torch.Tensor, float] = None,
               lorentz: tp.Union[torch.Tensor, float] = None
               ):
        raise NotImplementedError

    @property
    def config_shape(self):
        return self.modified_spin_system.config_shape

    def build_electron_electron(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        F = torch.zeros((*self.config_shape, self.modified_spin_system.dim, self.modified_spin_system.dim), dtype=torch.complex64,
                        device=self.modified_spin_system.device)
        for e_idx_1, e_idx_2, interaction in self.modified_spin_system.electron_electron:
            interaction = interaction.tensor.to(torch.complex64)
            F += scalar_tensor_multiplication(
                self.modified_spin_system.operator_cache[e_idx_1],
                self.modified_spin_system.operator_cache[e_idx_2],
                interaction)
        return F

    def build_electron_nuclei(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        F = torch.zeros((*self.config_shape, self.modified_spin_system.dim, self.modified_spin_system.dim), dtype=torch.complex64,
                        device=self.modified_spin_system.device)
        for e_idx, n_idx, interaction in self.modified_spin_system.electron_nuclei:
            interaction = interaction.tensor.to(torch.complex64)
            F += scalar_tensor_multiplication(
                self.modified_spin_system.operator_cache[e_idx],
                self.modified_spin_system.operator_cache[len(self.modified_spin_system.electrons) + n_idx],
                interaction)
        return F

    def build_nuclei_nuclei(self) -> torch.Tensor:
        """Constructs the zero-field Hamiltonian F."""
        F = torch.zeros((*self.config_shape, self.modified_spin_system.dim, self.modified_spin_system.dim), dtype=torch.complex64,
                        device=self.modified_spin_system.device)
        for n_idx_1, n_idx_2, interaction in self.modified_spin_system.nuclei_nuclei:
            interaction = interaction.tensor.to(torch.complex64)
            F += scalar_tensor_multiplication(
                self.modified_spin_system.operator_cache[len(self.modified_spin_system.electrons) + n_idx_1],
                self.modified_spin_system.operator_cache[len(self.modified_spin_system.electrons) + n_idx_2],
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
        G = torch.zeros((*self.config_shape, 3, self.modified_spin_system.dim, self.modified_spin_system.dim),
                        dtype=torch.complex64,
                        device=self.modified_spin_system.device)
        for idx, g_tensor in enumerate(self.modified_spin_system.g_tensors):
            g = g_tensor.tensor.to(torch.complex64)
            G += transform_tensor_components(self.modified_spin_system.operator_cache[idx], g)
        G *= (constants.BOHR / constants.PLANCK)
        return G

    def _build_nucleus_zeeman_terms(self) -> torch.Tensor:
        """Constructs the Nucleus interaction terms Gx, Gy, Gz. for nucleus spins"""
        G = torch.zeros((*self.config_shape, 3, self.modified_spin_system.dim, self.modified_spin_system.dim),
                        dtype=torch.complex64,
                        device=self.modified_spin_system.device)
        for idx, nucleus in enumerate(self.modified_spin_system.nuclei):
            g = nucleus.g_factor
            G += self.modified_spin_system.operator_cache[len(self.modified_spin_system.electrons) + idx] * g
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
        for idx, electron in enumerate(self.modified_spin_system.electrons):
            electron_contrib += electron.spin * torch.sum(
                self.modified_spin_system.g_tensors[idx].tensor[..., :, 0], dim=-1, keepdim=True).abs()

        nuclei_contrib = 0
        for idx, nucleus in enumerate(self.modified_spin_system.nuclei):
            nuclei_contrib += nucleus.spin * nucleus.g_factor.abs()

        return (electron_contrib * (constants.BOHR / constants.PLANCK) +
            nuclei_contrib * (constants.NUCLEAR_MAGNETRON / constants.PLANCK)).squeeze()

    def get_hamiltonian_terms(self) -> tuple:
        """
        Returns F, Gx, Gy, Gz.
        F is magnetic field free term
        Gx, Gy, Gz are terms multiplied to Bx, By, Bz respectively

        """
        return self.build_zero_field_term(), *self.build_zeeman_terms()

    def build_field_dep_straine(self):
        """
        Calculate electron Zeeman field dependant strained part
        :return:
        """
        for idx, g_tensor in enumerate(self.modified_spin_system.g_tensors):
            g = g_tensor.strained_tensor
            if g is None:
                pass
            else:
                g = g.to(torch.complex64)
                yield (
                    g_tensor.strain_correlation.to(torch.complex64),
                    self.modified_spin_system.operator_cache[idx],
                    g[..., :, 2, :] * constants.BOHR / constants.PLANCK)

    def build_zero_field_straine(self) -> torch.Tensor:
        """Constructs the zero-field strained part."""
        yield from self.build_electron_nuclei_straine()
        yield from self.build_electron_electron_straine()

    def build_electron_nuclei_straine(self) -> torch.Tensor:
        """Constructs the nuclei strained part."""
        for e_idx, n_idx, electron_nuclei_interaction in self.modified_spin_system.electron_nuclei:
            electron_nuclei = electron_nuclei_interaction.strained_tensor
            if electron_nuclei is None:
                pass
            else:
                electron_nuclei = electron_nuclei.to(torch.complex64)
                yield (
                    electron_nuclei_interaction.strain_correlation.to(torch.complex64),
                    self.modified_spin_system.operator_cache[e_idx],
                    self.modified_spin_system.operator_cache[len(self.modified_spin_system.electrons) + n_idx],
                    electron_nuclei)

    def build_electron_electron_straine(self) -> torch.Tensor:
        """Constructs the electron-electron strained part."""
        for e_idx_1, e_idx_2, electron_electron_interaction in self.modified_spin_system.electron_electron:
            electron_electron = electron_electron_interaction.strained_tensor
            if electron_electron is None:
                pass
            else:
                electron_electron = electron_electron.to(torch.complex64)
                yield (
                    electron_electron_interaction.strain_correlation.to(torch.complex64),
                    self.modified_spin_system.operator_cache[e_idx_1],
                    self.modified_spin_system.operator_cache[e_idx_2],
                    electron_electron)

    def __repr__(self):
        spin_system_summary = str(self.base_spin_system)

        lines = []
        lines.append(spin_system_summary)
        lines.append("\n" + "=" * 60)
        lines.append("GENERAL INFO: ")
        lines.append("=" * 60)

        lines.append(f"lorentz: {self.lorentz.item():.5f} T")
        lines.append(f"gauss: {self.gauss.item():.5f} T")
        ham_str = self.base_ham_strain
        ham_components = [f"{val:.4e}" if abs(val) >= 1e4 else f"{val:.4f}"
                         for val in ham_str.tolist()]
        lines.append(f"ham_str: {ham_components} Hz")

        return '\n'.join(lines)


class SpinSystemOrientator:
    """
    The helper class that allow toa
    transform spin system to spin system at different rotation angles.
    Effectively rotate all Hamiltonian parts
    """
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

    def _apply_reverse_transform(self, spin_system: SpinSystem, new_interactions: list[MultiOrientedInteraction]):
        # Determine how many interactions belong to each original group.
        num_g = len(spin_system.g_tensors)
        num_nuc = len(spin_system.electron_nuclei)

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
        #rotation_matrices = rotation_matrices.to(torch.complex64)
        config_shape = torch.Size([*spin_system.config_shape, rotation_matrices.shape[0]])
        interactions = [g_tensor for g_tensor in spin_system.g_tensors] + \
                       [el_nuc[-1] for el_nuc in spin_system.electron_nuclei] +\
                       [el_el[-1] for el_el in spin_system.electron_electron]
        strain_correlations = [g_tensor.strain_correlation for g_tensor in spin_system.g_tensors] + \
                       [el_nuc[-1].strain_correlation for el_nuc in spin_system.electron_nuclei] +\
                       [el_el[-1].strain_correlation for el_el in spin_system.electron_electron]

        interactions_tensors, strained_tensors = self.interactions_to_multioriented(interactions, rotation_matrices)
        interactions =\
            [
                MultiOrientedInteraction(interactions_tensor, strained_tensor, config_shape, strain_correlation) for
                interactions_tensor, strained_tensor, strain_correlation in
                zip(interactions_tensors, strained_tensors, strain_correlations)
            ]

        spin_system = self._apply_reverse_transform(spin_system, interactions)
        return spin_system


class MultiOrientedSample(BaseSample):
    def __init__(self, spin_system: SpinSystem,
                 ham_strain: tp.Optional[torch.Tensor] = None,
                 gauss: torch.Tensor = None,
                 lorentz: torch.Tensor = None,
                 mesh: tp.Optional[BaseMesh] = None,
                 ):
        """
        :param spin_system:
        SpinSystem
            The spin system describing electrons, nuclei, and their interactions.

        :param ham_strain:
        torch.Tensor, optional
            Anisotropic line width, due to the unresolved hyperfine interactions.
            The tensor components, provided in one of the following forms:
              - A scalar (for isotropic interaction).
              - A sequence of two values (axial and z components).
              - A sequence of three values (principal components).

        :param gauss:
        torch.Tensor, optional
            Gaussian broadening parameter(s). Defines inhomogeneous linewidth
            contributions (e.g., due to static disorder). Default is `None`.

        :param lorentz:
        torch.Tensor, optional
            Lorentzian broadening parameter(s). Defines homogeneous linewidth
            contributions (e.g., due to relaxation). Default is `None`

        :param mesh: The mesh to perform rotations.
        If it is None it wiil be initialize as DelaunayMeshNeighbour with initialsize = 20
        """

        super().__init__(spin_system, ham_strain, gauss, lorentz)
        self.mesh = self._init_mesh(mesh)
        rotation_matrices = self.mesh.rotation_matrices

        self._ham_strain = self._expand_hamiltonian(
            self.base_ham_strain,
            self.orientation_vector(rotation_matrices)
        )
        self.modified_spin_system = SpinSystemOrientator()(spin_system, rotation_matrices)

    def _init_mesh(self, mesh: tp.Optional[BaseMesh]):
        if mesh is None:
            mesh = mesher.DelaunayMeshNeighbour()
        return mesh

    def _expand_hamiltonian(self, ham_strain: torch.Tensor, orientation_vector: torch.Tensor):
        return torch.einsum("...i, ri -> ...r", ham_strain**2, orientation_vector**2).sqrt()

    def orientation_vector(self, rotation_matrices: torch.Tensor):
        return rotation_matrices[..., -1, :]

    def build_ham_strain(self) -> torch.Tensor:
        """Constructs the zero-field strained part of Hamiltonian"""
        return self._ham_strain

    def update(self,
               g_tensors: list[BaseInteraction] = None,
               electron_nuclei: list[tuple[int, int, BaseInteraction]] | None = None,
               electron_electron: list[tuple[int, int, BaseInteraction]] | None = None,
               nuclei_nuclei: list[tuple[int, int, BaseInteraction]] | None = None,
               ham_strain: tp.Optional[torch.Tensor] = None,
               gauss: tp.Union[torch.Tensor, float] = None,
               lorentz: tp.Union[torch.Tensor, float] = None
               ):
        """
        Update the parameters of a sample. No recomputation of spin vectors does not occur
        :param g_tensors:
        list[BaseInteraction]
            g-tensors corresponding to each electron in `electrons`.
            Each element must be an instance of `BaseInteraction` (e.g., `Interaction`).

        :param electron_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Hyperfine interactions between electrons and nuclei.
            Each tuple is of the form (electron_index, nucleus_index, interaction_tensor).
            Default is `None`.

        :param electron_electron:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or exchange interactions between pairs of electrons.
            Each tuple is of the form (electron_index, electron_index, interaction_tensor).
            Default is `None`.

        :param nuclei_nuclei:
        list[tuple[int, int, BaseInteraction]], optional
            Dipolar or J-coupling interactions between pairs of nuclei.
            Each tuple is of the form (nucleus_index, nucleus_index, interaction_tensor).
            Default is `None`

        :param ham_strain:
        torch.Tensor, optional
            Anisotropic line width, due to the unresolved hyperfine interactions.
            The tensor components, provided in one of the following forms:
              - A scalar (for isotropic interaction).
              - A sequence of two values (axial and z components).
              - A sequence of three values (principal components).

        :param gauss:
        torch.Tensor, optional
            Gaussian broadening parameter(s). Defines inhomogeneous linewidth
            contributions (e.g., due to static disorder). Default is `None`.

        :param lorentz:
        torch.Tensor, optional
            Lorentzian broadening parameter(s). Defines homogeneous linewidth
            contributions (e.g., due to relaxation). Default is `None`
        """

        rotation_matrices = self.mesh.rotation_matrices
        self.base_spin_system.update(g_tensors, electron_nuclei, electron_electron, nuclei_nuclei)

        if ham_strain is not None:
            self.base_ham_strain = self._init_ham_str(ham_strain)
            self._ham_strain = self._expand_hamiltonian(
                self.base_ham_strain,
                self.orientation_vector(rotation_matrices)
            )

        if gauss is None:
            self.gauss = torch.tensor(0.0, device=self.base_spin_system.device)
        else:
            self.gauss = torch.tensor(gauss)

        if lorentz is None:
            self.lorentz = torch.tensor(0.0, device=self.base_spin_system.device)
        else:
            self.lorentz = torch.tensor(lorentz)

        self.modified_spin_system = SpinSystemOrientator()(self.base_spin_system, rotation_matrices)
