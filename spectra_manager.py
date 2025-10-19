import math
import warnings
import typing as tp
from dataclasses import dataclass
from abc import ABC, abstractmethod

from spectral_integration import BaseSpectraIntegrator,\
    SpectraIntegratorEasySpinLike,\
    SpectraIntegratorEasySpinLikeTimeResolved, MeanIntegrator, AxialSpectraIntegratorEasySpinLike
from population import BaseTimeDependantPopulator, StationaryPopulator

import torch
import torch.fft as fft
import torch.nn as nn

import constants
import mesher
import res_field_algorithm
import res_freq_algorithm
import spin_system



def compute_matrix_element(vector_down: torch.Tensor, vector_up: torch.Tensor, G: torch.Tensor):
    tmp = torch.matmul(G.unsqueeze(-3), vector_down.unsqueeze(-1))
    return (vector_up.conj() * tmp.squeeze(-1)).sum(dim=-1)


class PostSpectraProcessing(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        :param gauss: The gauss parameter. The shape is [batch_size] or []
        :param lorentz: The lorentz parameter. The shape is [batch_size] or []
        """
        super().__init__()
        pass

    def _skip_broader(self, gauss, lorentz, magnetic_fields: torch.Tensor, spec: torch.Tensor):
        return spec

    def _broading_fabric(self, gauss: torch.Tensor, lorentz: torch.Tensor):
        # Check if all values are zero (not just any)
        gauss_zero = (gauss == 0).all()
        lorentz_zero = (lorentz == 0).all()

        if gauss_zero and lorentz_zero:
            return self._skip_broader
        elif not gauss_zero and lorentz_zero:
            return self._gauss_broader
        elif gauss_zero and not lorentz_zero:
            return self._lorentz_broader
        else:
            return self._voigt_broader

    def forward(self, gauss: torch.Tensor, lorentz: torch.Tensor,
                magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param gauss: Tensor of shape [] or [*batch_dims]
        :param lorentz: Tensor of shape [] or [*batch_dims]
        :param magnetic_field: Tensor of shape [N] or [*batch_dims, N]
        :param spec: Spectrum tensor of shape [N] or [*batch_dims, N]
        :return: Broadened spectrum, same shape as spec
        """
        squeeze_output = False
        if gauss.dim() == 0:
            gauss = gauss.unsqueeze(0)
        if lorentz.dim() == 0:
            lorentz = lorentz.unsqueeze(0)
        if magnetic_field.dim() == 1:
            magnetic_field = magnetic_field.unsqueeze(0)
            squeeze_output = True
        if spec.dim() == 1:
            spec = spec.unsqueeze(0)

        _broading_method = self._broading_fabric(gauss, lorentz)
        result = _broading_method(gauss, lorentz, magnetic_field, spec)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _build_lorentz_kernel(self, magnetic_field: torch.Tensor, fwhm_lorentz: torch.Tensor):
        """
        :param magnetic_field: Shape [*batch_dims, N]
        :param fwhm_lorentz: Shape [*batch_dims]
        :return: Kernel of shape [*batch_dims, N]
        """
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        N = magnetic_field.shape[-1]
        idx = torch.arange(N, device=magnetic_field.device) - N // 2

        # Reshape for broadcasting: idx -> [1, ..., 1, N]
        batch_dims = magnetic_field.dim() - 1
        idx_shape = [1] * batch_dims + [N]
        idx = idx.view(*idx_shape)

        # dH and fwhm_lorentz -> [*batch_dims, 1]
        dH = dH.unsqueeze(-1)
        gamma = (fwhm_lorentz.unsqueeze(-1) / 2)

        x = idx * dH
        L = (gamma / torch.pi) / (x ** 2 + gamma ** 2)
        return L

    def _build_gauss_kernel(self, magnetic_field: torch.Tensor, fwhm_gauss: torch.Tensor):
        """
        :param magnetic_field: Shape [*batch_dims, N]
        :param fwhm_gauss: Shape [*batch_dims]
        :return: Kernel of shape [*batch_dims, N]
        """
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        N = magnetic_field.shape[-1]
        idx = torch.arange(N, device=magnetic_field.device) - N // 2

        # Reshape for broadcasting: idx -> [1, ..., 1, N]
        batch_dims = magnetic_field.dim() - 1
        idx_shape = [1] * batch_dims + [N]
        idx = idx.view(*idx_shape)

        # dH and fwhm_gauss -> [*batch_dims, 1]
        dH = dH.unsqueeze(-1)
        sigma = fwhm_gauss.unsqueeze(-1) / (2 * (2 * torch.log(torch.tensor(2.0, device=magnetic_field.device))) ** 0.5)

        x = idx * dH
        G = torch.exp(-0.5 * (x / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)
        return G

    def _build_voigt_kernel(self,
                            magnetic_field: torch.Tensor,
                            fwhm_gauss: torch.Tensor,
                            fwhm_lorentz: torch.Tensor):
        """
        :param magnetic_field: Shape [*batch_dims, N]
        :param fwhm_gauss: Shape [*batch_dims]
        :param fwhm_lorentz: Shape [*batch_dims]
        :return: Kernel of shape [*batch_dims, N]
        """
        N = magnetic_field.shape[-1]
        G = self._build_gauss_kernel(magnetic_field, fwhm_gauss)
        L = self._build_lorentz_kernel(magnetic_field, fwhm_lorentz)

        Gf = fft.rfft(torch.fft.ifftshift(G, dim=-1), dim=-1)
        Lf = fft.rfft(torch.fft.ifftshift(L, dim=-1), dim=-1)

        Vf = Gf * Lf
        V = torch.fft.fftshift(fft.irfft(Vf, n=N, dim=-1), dim=-1)
        V = V / V.sum(dim=-1, keepdim=True)
        return V

    def _apply_convolution(self, spec: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution via FFT.
        :param spec: Shape [*batch_dims, N]
        :param kernel: Shape [*batch_dims, N]
        :return: Convolved spectrum of shape [*batch_dims, N]
        """
        S = fft.rfft(spec, dim=-1)
        K = fft.rfft(torch.fft.ifftshift(kernel, dim=-1), dim=-1)
        out = fft.irfft(S * K, n=spec.shape[-1], dim=-1)
        return out

    def _gauss_broader(self,
                       gauss: torch.Tensor, lorentz: torch.Tensor,
                       magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param gauss: Shape [*batch_dims]
        :param magnetic_field: Shape [*batch_dims, N]
        :param spec: Shape [*batch_dims, N]
        """
        kernel = self._build_gauss_kernel(magnetic_field, gauss)
        return self._apply_convolution(spec, kernel)

    def _lorentz_broader(self,
                         gauss: torch.Tensor, lorentz: torch.Tensor,
                         magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param lorentz: Shape [*batch_dims]
        :param magnetic_field: Shape [*batch_dims, N]
        :param spec: Shape [*batch_dims, N]
        """
        kernel = self._build_lorentz_kernel(magnetic_field, lorentz)
        return self._apply_convolution(spec, kernel)

    def _voigt_broader(self, gauss: torch.Tensor, lorentz: torch.Tensor,
                       magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param gauss: Shape [*batch_dims]
        :param lorentz: Shape [*batch_dims]
        :param magnetic_field: Shape [*batch_dims, N]
        :param spec: Shape [*batch_dims, N]
        """
        kernel = self._build_voigt_kernel(magnetic_field, gauss, lorentz)
        return self._apply_convolution(spec, kernel)


class IntegrationProcessorBase(nn.Module, ABC):
    def __init__(self,
                 mesh: mesher.BaseMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 chunk_size: int = 128,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer("threshold", torch.tensor(1e-4, device=device))
        self.spectra_integrator = self._init_spectra_integrator(spectra_integrator, harmonic,
                                                                chunk_size=chunk_size, device=device, dtype=dtype)
        self.mesh = mesh
        self.post_spectra_processor = post_spectra_processor
        self.to(device)

    @abstractmethod
    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int,
                                 chunk_size: int, device: torch.device, dtype: torch.dtype):
        pass

    @abstractmethod
    def _compute_areas(self, expanded_size: torch.Tensor, device: torch.device):
        pass

    @abstractmethod
    def _transform_data_to_mesh_format(
            self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor) -> \
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param res_fields: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :param intensities: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :param width: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :return:
        res_fields tensor with the resonance field at each triangle vertices. The shape is [..., 3] or [...]
        width tensor with the resonance field at each triangle vertices. The shape is [...]
        intensities tensor with the resonance field at each triangle vertices. The shape is [...]
        areas tensor with the resonance field at each triangle vertices. The shape is [...]
        """
        pass

    def _final_mask(self, res_fields: torch.Tensor, width: torch.Tensor,
                    intensities: torch.Tensor, areas: torch.Tensor):
        max_intensity = torch.amax(abs(intensities), dim=-1, keepdim=True)
        mask = ((intensities / max_intensity).abs() > self.threshold).any(dim=tuple(range(intensities.dim() - 1)))
        intensities = intensities[..., mask]
        width = width[..., mask]
        res_fields = res_fields[..., mask, :]
        areas = areas[..., mask]
        return res_fields, width, intensities, areas

    def forward(self,
                 res_fields: torch.Tensor,
                 intensities: torch.Tensor,
                 width: torch.Tensor,
                 gauss: torch.Tensor,
                 lorentz: torch.Tensor,
                 fields: torch.Tensor):

        res_fields, width, intensities, areas = (
            self._transform_data_to_mesh_format(
                res_fields, intensities, width
            )
        )
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)
        spec = self.spectra_integrator(
            res_fields, width, intensities, areas, fields
        )
        return self.post_spectra_processor(gauss, lorentz, fields, spec)


class IntegrationProcessorPowder(IntegrationProcessorBase):
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 chunk_size: int = 128,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32
                 ):
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         chunk_size=chunk_size, device=device, dtype=dtype
                         )

    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                 harmonic: int, chunk_size: int, device: torch.device, dtype: torch.dtype):
        if spectra_integrator is None:
            return SpectraIntegratorEasySpinLike(harmonic, chunk_size=chunk_size, device=device, dtype=dtype)
        else:
            return spectra_integrator

    def _compute_areas(self, expanded_size: torch.Tensor, device: torch.device):
        grid, simplices = self.mesh.post_mesh
        areas = self.mesh.spherical_triangle_areas(grid, simplices)
        areas = areas.reshape(1, -1).expand(expanded_size, -1).flatten()
        return areas

    def _process_tensor(self, data_tensor: torch.Tensor):
        _, simplices = self.mesh.post_mesh
        processed = self.mesh(data_tensor.transpose(-1, -2))
        return self.mesh.to_delaunay(processed, simplices)

    def _compute_batched_tensors(self, *args):
        batched_matrix = torch.stack(args, dim=-3)
        batched_matrix = self._process_tensor(batched_matrix)
        return batched_matrix

    def _transform_data_to_mesh_format(
            self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor) -> \
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param res_fields: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :param intensities: the tensor of resonance fields. The shape is [time, ..., num_resonance fields]
        :param width: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :return:
        res_fields tensor with the resonance field at each triangle vertices. The shape is [..., 3]
        width tensor with the resonance field at each triangle vertices. The shape is [...]
        intensities tensor with the resonance field at each triangle vertices. The shape is [...]
        areas tensor with the resonance field at each triangle vertices. The shape is [...]
        """
        batched_matrix = self._compute_batched_tensors(res_fields, intensities, width)
        expanded_size = batched_matrix.shape[-3]
        batched_matrix = batched_matrix.flatten(-3, -2)
        res_fields, intensities, width = torch.unbind(batched_matrix, dim=-3)

        width = width.mean(dim=-1)
        intensities = intensities.mean(dim=-1)
        areas = self._compute_areas(expanded_size, device=res_fields.device)
        return res_fields, width, intensities, areas


class IntegrationProcessorCrystal(IntegrationProcessorBase):
    def __init__(self,
                 mesh: mesher.CrystalMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 chunk_size: int = 128,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         chunk_size=chunk_size, device=device, dtype=dtype)

    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int,
                                 chunk_size: int, device: torch.device, dtype: torch.dtype):
        if spectra_integrator is None:
            return MeanIntegrator(harmonic, chunk_size=chunk_size, device=device)
        else:
            return spectra_integrator

    def _compute_areas(self, expanded_size: torch.Size, device: torch.device):
        areas = torch.ones(expanded_size, dtype=torch.float32, device=device)
        return areas

    def _final_mask(self, res_fields: torch.Tensor, width: torch.Tensor,
                    intensities: torch.Tensor, areas: torch.Tensor):
        max_intensity = torch.amax(abs(intensities), dim=-1, keepdim=True)
        mask = ((intensities / max_intensity).abs() > self.threshold).any(dim=tuple(range(intensities.dim() - 1)))

        intensities = intensities[..., mask]
        width = width[..., mask]
        res_fields = res_fields[..., mask]
        areas = areas[..., mask]
        return res_fields, width, intensities, areas

    def _transform_data_to_mesh_format(
            self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor) -> \
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param res_fields: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :param intensities: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :param width: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :return:
        res_fields tensor with the resonance field at each triangle vertices. The shape is [...]
        width tensor with the resonance field at each triangle vertices. The shape is [...]
        intensities tensor with the resonance field at each triangle vertices. The shape is [...]
        areas tensor with the resonance field at each triangle vertices. The shape is [...]
        """
        res_fields = res_fields.flatten(-2, -1)
        intensities = intensities.flatten(-2, -1)
        width = width.flatten(-2, -1)

        expanded_size = res_fields.shape
        areas = self._compute_areas(expanded_size, res_fields.device)
        return res_fields, width, intensities, areas


class IntegrationProcessorTimeResolved(IntegrationProcessorPowder):
    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int,
                                 chunk_size: int, device: torch.device, dtype: torch.dtype):
        if spectra_integrator is None:
            return SpectraIntegratorEasySpinLikeTimeResolved(
                harmonic, chunk_size=chunk_size,
                device=device, dtype=dtype
            )
        else:
            return spectra_integrator

    def _compute_areas(self, expanded_size: torch.Tensor, device: torch.device):
        grid, simplices = self.mesh.post_mesh
        areas = self.mesh.spherical_triangle_areas(grid, simplices)
        areas = areas.reshape(1, -1).expand(expanded_size, -1).flatten()
        return areas

    def _transform_data_to_mesh_format(self, res_fields: torch.Tensor,
                                       intensities: torch.Tensor,
                                       width: torch.Tensor):
        """
        :param res_fields: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :param intensities: the tensor of resonance fields. The shape is [time_dim, ..., num_resonance fields]
        :param width: the tensor of resonance fields. The shape is [..., num_resonance fields]
        :return:
        res_fields tensor with the resonance field at each triangle vertices. The shape is [..., 3]
        width tensor with the resonance field at each triangle vertices. The shape is [...]
        intensities tensor with the resonance field at each triangle vertices. The shape is [time_dim, ...]
        areas tensor with the resonance field at each triangle vertices. The shape is [...]
        """
        batched_matrix = self._compute_batched_tensors(res_fields, width)
        expanded_size = batched_matrix.shape[-3]
        batched_matrix = batched_matrix.flatten(-3, -2)
        intensities = self._process_tensor(intensities)
        intensities = intensities.flatten(-3, -2)

        res_fields, width = torch.unbind(batched_matrix, dim=-3)
        width = width.mean(dim=-1)
        intensities = intensities.mean(dim=-1)
        areas = self._compute_areas(expanded_size, device=res_fields.device)
        return res_fields, width, intensities, areas


class Broadener(nn.Module):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.register_buffer("_width_conversion", torch.tensor(1 / math.sqrt(2 * math.log(2)), device=device))
        self.to(device)

    def _compute_element_field_free(self, vector: torch.Tensor,
                          tensor_components_A: torch.Tensor, tensor_components_B: torch.Tensor,
                          transformation_matrix: torch.Tensor, correlation_matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            '...pij,jkl,ikl,...bk,...bl,ph->...hb',
            transformation_matrix, tensor_components_A, tensor_components_B, torch.conj(vector), vector,
            correlation_matrix
        ).real

    def _compute_element_field_dep(self, vector: torch.Tensor,
                          tensor_components: torch.Tensor,
                          transformation_matrix: torch.Tensor, correlation_matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            '...pi, ikl,...bk,...bl,ph->...hb',
            transformation_matrix, tensor_components, torch.conj(vector), vector, correlation_matrix
        ).real

    def _compute_field_straine_square(self, strained_data, vector_down, vector_up, B_trans):
        correlation_matrix, tensor_components, transformation_matrix = strained_data

        return (B_trans.unsqueeze(-2) * (
                self._compute_element_field_dep(vector_up, tensor_components, transformation_matrix,
                                                correlation_matrix) -
                self._compute_element_field_dep(vector_down, tensor_components, transformation_matrix,
                                                correlation_matrix)
        )).square().sum(dim=-2)

    def _compute_field_free_straine_square(self, strained_data, vector_down, vector_up):
        correlation_matrix, tensor_components_A, tensor_components_B, transformation_matrix = strained_data
        return (
                self._compute_element_field_free(
                    vector_up, tensor_components_A, tensor_components_B, transformation_matrix, correlation_matrix
                ) -
                self._compute_element_field_free(
                    vector_down, tensor_components_A, tensor_components_B, transformation_matrix, correlation_matrix
                )
        ).square().sum(dim=-2)

    def add_hamiltonian_straine(self, sample: spin_system.MultiOrientedSample, squared_width):
        hamiltonian_width = sample.build_ham_strain().unsqueeze(-1).square()
        return (squared_width + hamiltonian_width).sqrt()

    def forward(self, sample: spin_system.MultiOrientedSample,
                 vector_down: torch.Tensor, vector_up: torch.Tensor, B_trans: torch.Tensor):
        target_shape = vector_down.shape[:-1]
        result = torch.zeros(target_shape, dtype=B_trans.dtype, device=vector_down.device)

        for strained_data in sample.build_field_dep_straine():
            result += self._compute_field_straine_square(strained_data, vector_down, vector_up, B_trans)

        for strained_data in sample.build_zero_field_straine():
            result += self._compute_field_free_straine_square(strained_data, vector_down, vector_up)

        return self.add_hamiltonian_straine(sample, result) * self._width_conversion  # To convert from p-p into width and half height


class BaseIntensityCalculator(nn.Module):
    def __init__(self, spin_system_dim: int | list[int],
                 temperature: tp.Optional[float] = None,
                 populator: tp.Optional[tp.Callable] = None,
                 tolerancy: float = 1e-12, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.register_buffer("tolerancy", torch.tensor(tolerancy, device=device))
        self.populator = self._init_populator(populator, temperature, device)
        self.spin_system_dim = spin_system_dim
        self.temperature = temperature
        self.to(device)

    def _init_populator(self, populator: tp.Optional[tp.Callable], temperature: tp.Optional[float],
                        device: torch.device):
        return populator

    def _compute_magnitization_part(self, Gx, Gy, Gz, vector_down, vector_up):
        magnitization = compute_matrix_element(vector_down, vector_up, Gx).square().abs() + \
                        compute_matrix_element(vector_down, vector_up, Gy).square().abs()
        return magnitization * (constants.PLANCK / constants.BOHR) ** 2

    def compute_intensity(self, Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies,
                          resonance_manifold, full_system_vectors, *args, **kwargs):
        raise NotImplementedError

    def forward(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                vector_down: torch.Tensor, vector_up: torch.Tensor, lvl_down: torch.Tensor,
                lvl_up: torch.Tensor, resonance_energies: torch.Tensor, resonance_manifold,
                full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs):
        return self.compute_intensity(Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies,
                                      resonance_manifold, full_system_vectors)


class StationaryIntensitiesCalculator(BaseIntensityCalculator):
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float] = None,
                 populator: tp.Optional[tp.Callable] = None, tolerancy: float = 1e-14,
                 device: torch.device = torch.device("cpu")):
        super().__init__(spin_system_dim, temperature, populator, tolerancy, device=device)

    def _init_populator(self, populator, temperature, device: torch.device):
        if populator is None:
            return StationaryPopulator(temperature, device=device)
        else:
            return populator

    def compute_intensity(self, Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies,
                          resonance_manifold, full_system_vectors: tp.Optional[torch.Tensor], *args, **kwargs):
        """Base method to compute intensity (to be overridden)."""
        intensity = self.populator(resonance_energies, lvl_down, lvl_up, *args, **kwargs) * (
                self._compute_magnitization_part(Gx, Gy, Gz, vector_down, vector_up)
        )
        return intensity


class TimeResolvedIntensitiesCalculator(BaseIntensityCalculator):
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float],
                 populator: tp.Optional[BaseTimeDependantPopulator],
                 tolerancy=1e-14, device: torch.device = torch.device("cpu")
                 ):
        super().__init__(spin_system_dim, temperature, populator, tolerancy, device=device)

    def compute_intensity(self, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                          vector_down: torch.Tensor, vector_up: torch.Tensor,
                          lvl_down: torch.Tensor, lvl_up: torch.Tensor, resonance_energies: torch.Tensor,
                          resonance_manifold: torch.Tensor, full_system_vectors: tp.Optional[torch.Tensor],
                          *args, **kwargs):
        """
        :param Gx:
        :param Gy:
        :param Gz:
        :param vector_down:
        :param vector_up:
        :param lvl_down:
        :param lvl_up:
        :param resonance_energies:
        :param resonance_manifold: Resonance Values of magnetic field or resonance frequency
        :param full_system_vectors:
        :param args:
        :param kwargs:
        :return:
        """
        intensity = (
                self._compute_magnitization_part(Gx, Gy, Gz, vector_down, vector_up)
        )
        return intensity

    def calculate_population_evolution(self, time: torch.Tensor,
                                       res_fields, lvl_down, lvl_up,
                                       resonance_energies, vector_down, vector_up,
                                       *args, **kwargs):

        return self.populator(time, res_fields, lvl_down,
                              lvl_up, resonance_energies, vector_down, vector_up, *args, **kwargs)


class MultiSampleIntensitiesCalculator(BaseIntensityCalculator):
    def __init__(self,
                 spin_system_dim: int | list[int],
                 temperature: tp.Optional[float],
                 populator: BaseTimeDependantPopulator,
                 tolerancy=1e-12, device: torch.device = torch.device("cpu")
                 ):
        super().__init__(spin_system_dim, temperature, populator, tolerancy, device=device)

    def calculate_population_evolution(self, time: torch.Tensor, intensity_outs, spin_dimensions: list[int]):
        populations = self.populator(time, intensity_outs, spin_dimensions)
        return populations

@dataclass
class ParamSpec:
    """
    Let's consider the Hamiltonian with shape [..., N, N], where N is spin system size
    Its resonance fields have dimension [...., K]. Let's call it 'scalar'
    Its eigen values have dimension [..., K, N], where K is number of resonance transitions. Let's call it 'vector'
    Its eigen vectors have dimension [..., K, N, N], where K is number of resonance transitions. Let's call it 'matrix'

    For some purposes it is necessary to get not only intensities, res-fields and width at resonance points
    but other parameters. To generalize the approach of making these parameters it is necessary to te
    """
    category: str
    dtype: torch.dtype

    def __post_init__(self):
        assert self.category in (
            "scalar", "vector", "matrix"), f"Category must be one of 'scalar', 'vector', 'matrix', got {self.category}"


class BaseSpectraCreator(nn.Module, ABC):
    """
    Base clas of spectra creators
    """
    def __init__(self,
                 resonance_parameter: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[float] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[BaseIntensityCalculator] = None,
                 populator: tp.Optional[StationaryPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 integration_chunk_size: int = 128,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        :param resonance_parameter: Resonance parameter of experiment: frequency or field

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as default calculator specific to given spectra_creator

        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If it is None then it will be initialized as default calculator specific to given intensity_calculator

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance
        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it
        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For stationary creator is True, for time resolves is False

        :param integration_chunk_size:
            Chunk Size of integration process. Current implementation of powder integration is iterative.
            For whole set of resonance lines chunk size of spectral freq/field is computed.
            Increasing the size increases the integration speed, but also increases the required memory allocation.

        :param dtype: float32 / float64
        Base dtype for all types of operations. If complex parameters is used,
        they will be converted in complex64, complex128

        """
        super().__init__()
        self.register_buffer("resonance_parameter", torch.tensor(resonance_parameter, device=device, dtype=dtype))
        self.register_buffer("threshold", torch.tensor(1e-2, device=device, dtype=dtype))
        self.register_buffer("tolerancy", torch.tensor(1e-10, device=device, dtype=dtype))
        self.register_buffer("intensity_std", torch.tensor(1e-7, device=device, dtype=dtype))

        self.spin_system_dim, self.batch_dims, self.mesh =\
            self._init_sample_parameters(sample, spin_system_dim, batch_dims, mesh)
        self.mesh_size = self.mesh.initial_size
        self.broader = Broadener(device=device)

        self.res_algorithm = self._init_res_algorithm(device=device, dtype=dtype)

        self.intensity_calculator = self._get_intenisty_calculator(intensity_calculator,
                                                                   temperature, populator, device=device)
        self._param_specs = self._get_param_specs()

        self.spectra_processor = self._init_spectra_processor(spectra_integrator,
                                                              harmonic,
                                                              post_spectra_processor,
                                                              chunk_size=integration_chunk_size,
                                                              device=device, dtype=dtype)
        self.recompute_spin_parameters = recompute_spin_parameters
        self._init_cached_parameters()
        self.to(device)

    def _init_cached_parameters(self):
        if not self.recompute_spin_parameters:
            self._cashed_flag = False
            self.vectors_u = None
            self.vectors_v = None
            self.valid_lvl_down = None
            self.valid_lvl_up = None
            self.res_fields = None
            self.resonance_energies = None
            self.full_eigen_vectors = None
            self._resfield_method = self._cashed_resfield

        else:
            self._resfield_method = self._recomputed_resfield

    def _init_res_algorithm(self, device: torch.device, dtype: torch.dtype):
        return res_field_algorithm.ResField(
            spin_system_dim=self.spin_system_dim,
            mesh_size=self.mesh_size,
            batch_dims=self.batch_dims,
            output_full_eigenvector=self._get_output_eigenvector(),
            device=device,
            dtype=dtype
        )

    @abstractmethod
    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                chunk_size: int,
                                device: torch.device,
                                dtype: torch.dtype) -> IntegrationProcessorBase:
        pass

    def _init_sample_parameters(self,
                                sample: tp.Optional[spin_system.MultiOrientedSample],
                                spin_system_dim: tp.Optional[int],
                                batch_dims: tp.Optional[float],
                                mesh: tp.Optional[mesher.BaseMesh]):
        if sample is None:
            if (spin_system_dim is not None) and (batch_dims is not None) and (mesh is not None):
                return spin_system_dim, batch_dims, mesh
            else:
                raise TypeError("You should pass sample or spin_system_dim, batch_dims, mesh arguments")
        else:
            spin_system_dim = sample.base_spin_system.spin_system_dim
            batch_dims = sample.config_shape[:-1]
            mesh = sample.mesh

        return spin_system_dim, batch_dims, mesh

    def _get_intenisty_calculator(self, intensity_calculator, temperature: float, populator: StationaryPopulator,
                                  device:torch.device):
        if intensity_calculator is None:
            return StationaryIntensitiesCalculator(self.spin_system_dim, temperature, populator, device=device)
        else:
            return intensity_calculator

    def _freq_to_field(self, vector_down, vector_up, Gz):
        """Compute frequency-to-field contribution"""
        factor_1 = compute_matrix_element(vector_up, vector_up, Gz)
        factor_2 = compute_matrix_element(vector_down, vector_down, Gz)

        diff = (factor_1 - factor_2).abs()
        safe_diff = torch.where(diff < self.tolerancy, self.tolerancy, diff)
        return safe_diff.reciprocal()

    def _get_output_eigenvector(self) -> bool:
        return False

    def _get_param_specs(self) -> list[ParamSpec]:
        """
        :return: list[ParamSpec]. The number of parameters and
        their order must coincide with output of method _mask_additional
        """
        return []

    def _cashed_resfield(self, sample: spin_system.MultiOrientedSample,
                                B_low: torch.Tensor, B_high: torch.Tensor,
                                F: torch.Tensor, Gz: torch.Tensor):
        if not self._cashed_flag:
            (self.vectors_u, self.vectors_v), (self.valid_lvl_down, self.valid_lvl_up), self.res_fields, \
                self.resonance_energies, self.full_eigen_vectors = \
                self._recomputed_resfield(sample, B_low, B_high, F, Gz)

            self._cashed_flag = True

        return (self.vectors_u, self.vectors_v), (self.valid_lvl_down, self.valid_lvl_up), self.res_fields, \
            self.resonance_energies, self.full_eigen_vectors

    def _recomputed_resfield(self, sample: spin_system.MultiOrientedSample,
                                B_low: torch.Tensor, B_high: torch.Tensor,
                                F: torch.Tensor, Gz: torch.Tensor):
        (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up), res_fields, resonance_energies, full_eigen_vectors =\
                self.res_algorithm(sample, self.resonance_parameter, B_low, B_high, F, Gz)

        return (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up), res_fields,\
            resonance_energies, full_eigen_vectors

    def forward(self,
                 sample: spin_system.MultiOrientedSample,
                 fields: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """
        :param sample: MultiOrientedSample object
        :param fields: The magnetic fields in Tesla units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return:
        """
        B_low = fields[..., 0]
        B_high = fields[..., -1]
        B_low = B_low.unsqueeze(-1).repeat(*([1] * B_low.ndim), *self.mesh_size)
        B_high = B_high.unsqueeze(-1).repeat(*([1] * B_high.ndim), *self.mesh_size)

        F, Gx, Gy, Gz = sample.get_hamiltonian_terms()

        (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up), res_fields, \
            resonance_energies, full_eigen_vectors = self._resfield_method(sample, B_low, B_high, F, Gz)
        if (vectors_u.shape[-2] == 0):
            return torch.zeros_like(fields)
        res_fields, intensities, width, *extras = self.compute_parameters(sample, F, Gx, Gy, Gz,
                                              vectors_u, vectors_v,
                                              valid_lvl_down, valid_lvl_up,
                                              res_fields,
                                              resonance_energies,
                                              full_eigen_vectors)

        res_fields, intensities, width = self._postcompute_batch_data(
            res_fields, intensities, width, F, Gx, Gy, Gz, time, *extras, **kwargs
        )

        gauss = sample.gauss
        lorentz = sample.lorentz

        return self._finalize(res_fields, intensities, width, gauss, lorentz, fields)

    def _postcompute_batch_data(self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor, time: tp.Optional[torch.Tensor], *extras,  **kwargs):
        return res_fields, intensities, width

    def _finalize(self,
                  res_fields: torch.Tensor,
                  intensities: torch.Tensor,
                  width: torch.Tensor,
                  gauss: torch.Tensor,
                  lorentz: torch.Tensor,
                  fields: torch.Tensor):
        return self.spectra_processor(res_fields, intensities, width, gauss, lorentz, fields)

    def _mask_components(self, intensities_mask: torch.Tensor, *extras) -> list[tp.Any]:
        updated_extras = []
        for idx, param_spec in enumerate(self._param_specs):
            if param_spec.category == "scalar":
                updated_extras.append(extras[idx][..., intensities_mask])

            elif param_spec.category == "vector":
                updated_extras.append(extras[idx][..., intensities_mask, :])

            elif param_spec.category == "matrix":
                updated_extras.append(extras[idx][..., intensities_mask, :, :])
        return updated_extras

    def _mask_additional(self, vector_down: torch.Tensor, vector_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor],):
        return ()

    def _compute_additional(self,
                           sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor, *extras):
        return extras

    def compute_parameters(self, sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           vector_down: torch.Tensor, vector_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           res_fields: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor]) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tp.Any]]:
        """
        :param sample: The sample which transitions must be found
        :param F: Magnetic free part of spin Hamiltonian H = F + B * G
        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param resonance_energies:
            Energies of spin states. The shape is [..., N]

        :param res_fields: Resonance fields. The shape os [..., N]

        :param full_system_vectors: Eigen vector of each level of a spin system. The shape os [..., N, N]

        :return: tuple of the next data
         - Resonance fields
         - Intensities of transitions
         - Width of transition lines
         - extras parameters computed in _compute_additional
        """
        intensities = self.intensity_calculator.compute_intensity(
            Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies, res_fields, full_system_vectors
        )
        lines_dimension = tuple(range(intensities.ndim - 1))
        intensities_mask = (intensities / intensities.abs().max() > self.threshold).any(dim=lines_dimension)
        intensities = intensities[..., intensities_mask]

        extras = self._mask_additional(vector_down,
            vector_up, lvl_down, lvl_up, resonance_energies,
            full_system_vectors)

        extras = self._mask_components(intensities_mask, *extras)

        res_fields = res_fields[..., intensities_mask]
        vector_u = vector_down[..., intensities_mask, :]
        vector_v = vector_up[..., intensities_mask, :]

        freq_to_field = self._freq_to_field(vector_u, vector_v, Gz)
        intensities *= freq_to_field
        intensities = intensities / self.intensity_std
        width = self.broader(sample, vector_u, vector_v, res_fields) * freq_to_field

        extras = self._compute_additional(
            sample, F, Gx, Gy, Gz, *extras
        )

        return res_fields, intensities, width, *extras


class StationarySpectraCreator(BaseSpectraCreator):
    """
    Spectra Creator to compute CW spectra. As default it computes powder spectra.
    """
    def __init__(self,
                 freq: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[float] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[BaseIntensityCalculator] = None,
                 populator: tp.Optional[StationaryPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 integration_chunk_size: int = 128,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        :param freq: Resonance frequency of experiment

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as StationaryIntensitiesCalculator

        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If intensity_calculator is None or StationaryIntensitiesCalculator
            then it will be initialized as StationaryPopulation
            In this case the population is given as Boltzmann population

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 1.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For stationary creator is True.

        :param integration_chunk_size:
            Chunk Size of integration process. Current implementation of powder integration is iterative.
            For whole set of resonance lines chunk size of spectral freq/field is computed.
            Increasing the size increases the integration speed, but also increases the required memory allocation.
        """
        super().__init__(freq, sample, spin_system_dim, batch_dims, mesh, intensity_calculator,
                         populator, spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters,
                         integration_chunk_size=integration_chunk_size,
                         device=device, dtype=dtype)

    def _postcompute_batch_data(self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor,
                                time: tp.Optional[torch.Tensor],  *extras, **kwargs):
        return res_fields, intensities, width

    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                chunk_size: int,
                                device: torch.device,
                                dtype: torch.dtype) -> IntegrationProcessorBase:
        if self.mesh.name == "PowderMesh":
            return IntegrationProcessorPowder(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                              chunk_size=chunk_size, device=device, dtype=dtype)

        elif self.mesh.name == "CrystalMesh":
            return IntegrationProcessorCrystal(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                               chunk_size=chunk_size, device=device, dtype=dtype)

        else:
            return IntegrationProcessorPowder(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                              chunk_size=chunk_size, device=device, dtype=dtype)

    def __call__(self,
                sample: spin_system.MultiOrientedSample,
                fields: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """
        :param sample: MultiOrientedSample object
        :param fields: The magnetic fields in Tesla units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return:
        """
        return super().__call__(sample, fields, time)


class TruncatedSpectraCreatorTimeResolved(BaseSpectraCreator):
    """
    Time Resolved Spectra Creator.
    Unlike CoupledSpectraCreatorTimeResolved,
    it does not calculate eigenvectors for all states in the case of resonant fields

    Either Populator or Intensity Calculator should be given
    """
    def __init__(self,
                 freq: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[float] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[tp.Callable] = None,
                 populator: tp.Optional[StationaryPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 0,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 ):

        """
        Note that by default these spin systems (energies, vectors, etc.) are calculated once and then cached.
        Default harmoinc is None

        :param freq: Resonance frequency of experiment

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as TimeResolvedIntensitiesCalculator

        :param populator:
            Class that is used to compute part intensity due to population of levels.
            There is no default initialization.
            Eather Populator or Intensity Calculator should be given

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 0.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For time resolved spectra creator is False

        """
        super().__init__(freq, sample, spin_system_dim, batch_dims, mesh, intensity_calculator, populator,
                         spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters, device=device, dtype=dtype)

    def __call__(self, sample: spin_system.MultiOrientedSample, field: torch.Tensor, time: torch.Tensor, **kwargs) ->\
            torch.Tensor:
        """
        :param sample: MultiOrientedSample object
        :param fields: The magnetic fields in Tesla units
        :param time: Time to compute time resolved spectra
        :param kwargs:
        :return: EPR spectra
        """
        return super().__call__(sample, field, time, **kwargs)

    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int,
                                 chunk_size: int, device: torch.device, dtype: torch.dtype):
        if spectra_integrator is None:
            self.spectra_integrator = SpectraIntegratorEasySpinLikeTimeResolved(
                harmonic=harmonic,
                chunk_size=chunk_size,
                device=device,
                dtype=dtype)
        else:
            self.spectra_integrator = spectra_integrator

    def _get_intenisty_calculator(self, intensity_calculator,
                                  temperature,
                                  populator: tp.Optional[BaseTimeDependantPopulator], device: torch.device):
        if intensity_calculator is None:
            return TimeResolvedIntensitiesCalculator(self.spin_system_dim, temperature, populator, device=device)
        else:
            return intensity_calculator

    def _get_param_specs(self) -> list[ParamSpec]:
        params = [
            ParamSpec("scalar", torch.long),
            ParamSpec("scalar", torch.long),
            ParamSpec("vector", torch.float32),
            ParamSpec("vector", torch.complex64),
            ParamSpec("vector", torch.complex64)
            ]
        return params

    def _mask_additional(self, vector_down: torch.Tensor, vector_up: torch.Tensor,
                        lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                        resonance_energies: torch.Tensor,
                        full_system_vectors: tp.Optional[torch.Tensor]):

        return lvl_down, lvl_up, resonance_energies, vector_down, vector_up

    def _postcompute_batch_data(self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor, time: torch.Tensor, *extras, **kwargs):
        lvl_down, lvl_up, resonance_energies, vector_down, vectors_up, *extras = extras

        population = self.intensity_calculator.calculate_population_evolution(
            time, res_fields, lvl_down, lvl_up, resonance_energies, vector_down, vectors_up, *extras
        )
        intensities = (intensities.unsqueeze(0) * population)
        return res_fields, intensities, width

    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                chunk_size: int,
                                device: torch.device,
                                dtype: torch.dtype) -> IntegrationProcessorBase:
        if self.mesh.name == "PowderMesh":
            return IntegrationProcessorTimeResolved(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                                    chunk_size=chunk_size, device=device, dtype=dtype)
        else:
            return IntegrationProcessorTimeResolved(self.mesh, spectra_integrator, harmonic, post_spectra_processor,
                                                    chunk_size=chunk_size, device=device, dtype=dtype)

    def _init_recompute_spin_flag(self) -> bool:
        """
        If flag is False: resfield data is cached.
        If flag is True: resfield recomputes every time
        :return:
        """
        return False

    def update_context(self, new_conteext: tp.Any):
        self.intensity_calculator.populator.context = new_conteext


class CoupledSpectraCreatorTimeResolved(TruncatedSpectraCreatorTimeResolved):
    """
    Time Resolved Spectra Creator.
    Unlike TruncatedSpectraCreatorTimeResolved,
    it does calculate eigenvectors for all states in the case of resonant fields and pass it as args in populator
    full system vectors have dimension [...., Tr, N, N], where Tr is number of transitions N, N are spin system size.

    Either Populator or Intensity Calculator should be given
    """

    def _get_output_eigenvector(self) -> bool:
        return True

    def _get_param_specs(self) -> list[ParamSpec]:
        params = [
            ParamSpec("scalar", torch.long),
            ParamSpec("scalar", torch.long),

            ParamSpec("vector", torch.float32),
            ParamSpec("vector", torch.complex64),

            ParamSpec("vector", torch.complex64),
            ParamSpec("matrix", torch.complex64)
            ]
        return params

    def _mask_additional(self, vector_down: torch.Tensor, vector_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor]):

        return lvl_down, lvl_up, resonance_energies, vector_down, vector_up, full_system_vectors

    def _compute_additional(self, sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           *args):
        return args


class MultiSampleCreator(nn.Module):
    def __init__(self,
                 creators: list[CoupledSpectraCreatorTimeResolved],
                 freq: tp.Optional[tp.Union[float, torch.Tensor]] = None,
                 temperature: tp.Optional[float] = 293,
                 intensity_calculator: tp.Optional[MultiSampleIntensitiesCalculator] = None,
                 populator: tp.Optional[BaseTimeDependantPopulator] = None,
                 weights: tp.Optional[torch.Tensor] = None,
                 ):
        """
        creators[i] is already configured for sample i (its spin_system_dim, mesh, …).
        """
        super().__init__()
        self.resonance_parameter = torch.tensor(freq) if freq is not None else creators[0].resonance_parameter
        self.spin_system_dim = [creator.spin_system_dim for creator in creators]
        if len(creators) == 0:
            raise ValueError("Need at least one creator")
        self.creators = list(creators)
        self.intensity_calculator = self._get_intensity_calculator(intensity_calculator, temperature, populator)
        if weights is None:
            self.weights = torch.ones(len(creators), dtype=torch.float32)
        else:
            self.weights = weights
        self.mesh_size = self.creators[0].mesh_size

    def _get_intensity_calculator(self,
                                  intensity_calculator: tp.Optional[MultiSampleIntensitiesCalculator],
                                  temperature: float,
                                  populator: tp.Optional[BaseTimeDependantPopulator]):
        if intensity_calculator is None:
            return MultiSampleIntensitiesCalculator(self.spin_system_dim, temperature, populator)
        else:
            return intensity_calculator

    def _postcompute_intesities(self,
                                time: torch.Tensor,
                                intensities_samples: list[torch.Tensor],
                                population_samples: list[tuple]):
        intensities_samples_finile = []
        population_samples = self.intensity_calculator.calculate_population_evolution(time,
                                                                                      population_samples,
                                                                                      self.spin_system_dim)
        for intensities, population in zip(intensities_samples, population_samples):
            intensities_samples_finile.append(intensities.unsqueeze(0) * population)
        return intensities_samples_finile

    def forward(self,
                 samples: tp.Sequence[spin_system.MultiOrientedSample],
                 fields: torch.Tensor, time: torch.Tensor
                 ) -> torch.Tensor:
        if len(samples) != len(self.creators):
            raise ValueError(f"Expected {len(self.creators)} samples, got {len(samples)}")

        B_low = fields[..., 0]
        B_high = fields[..., -1]

        B_low = B_low.unsqueeze(-1).repeat(*([1] * B_low.ndim), *self.mesh_size)
        B_high = B_high.unsqueeze(-1).repeat(*([1] * B_high.ndim), *self.mesh_size)

        intensities_samples = []
        widths_samples = []
        population_samples = []
        for sample, creator in zip(samples, self.creators):
            F, Gx, Gy, Gz = sample.get_hamiltonian_terms()

            (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up), res_fields, \
                resonance_energies, full_eigen_vectors = creator._resfield_method(sample, B_low, B_high, F, Gz)

            res_fields, intensities, width, *extras = creator.compute_parameters(sample, F, Gx, Gy, Gz,
                                                                              vectors_u, vectors_v,
                                                                              valid_lvl_down, valid_lvl_up,
                                                                              res_fields,
                                                                              resonance_energies,
                                                                              full_eigen_vectors)

            lvl_down, lvl_up, resonance_energies, vector_down, vector_up, full_system_vectors = extras

            intensities_samples.append(intensities)
            widths_samples.append(width)

            population_samples.append((res_fields, lvl_down, lvl_up,
                                       resonance_energies, vector_down, vector_up,
                                       full_system_vectors, F, Gz))

        intensities_samples = self._postcompute_intesities(
            time, intensities_samples, population_samples
        )

        spectras = []
        for idx, creator in enumerate(self.creators):
            res_fields, intensities, width = (population_samples[idx][0],
                           intensities_samples[idx], widths_samples[idx])

            gauss = samples[idx].gauss
            lorentz = samples[idx].lorentz
            spectra = creator._finalize(res_fields, intensities, width, gauss, lorentz, fields)

            spectras.append(spectra)
        return torch.stack(spectras, dim=0)


class StationarySpectraCreatorFreq(StationarySpectraCreator):
    """
    Spectra Creator to compute CW spectra. As default it computes powder spectra.
    """
    def __init__(self,
                 field: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[float] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[BaseIntensityCalculator] = None,
                 populator: tp.Optional[StationaryPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = 293,
                 recompute_spin_parameters: bool = True,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32
                 ):
        """
        :param field: Resonance field of experiment

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as StationaryIntensitiesCalculator

        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If intensity_calculator is None or StationaryIntensitiesCalculator
            then it will be initialized as StationaryPopulation
            In this case the population is given as Boltzmann population

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum.

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance. Default is 1.

        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it

        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For stationary creator is True.

        """
        super().__init__(field, sample, spin_system_dim, batch_dims, mesh, intensity_calculator,
                         populator, spectra_integrator, harmonic, post_spectra_processor,
                         temperature, recompute_spin_parameters, device=device, dtype=dtype)

    def _init_res_algorithm(self, device, dtype: torch.dtype):
        return res_freq_algorithm.ResFreq(
            spin_system_dim=self.spin_system_dim,
            mesh_size=self.mesh_size,
            batch_dims=self.batch_dims,
            output_full_eigenvector=self._get_output_eigenvector(),
            device=device,
            dtype=dtype
        )

    def __call__(self,
                sample: spin_system.MultiOrientedSample,
                freq: torch.Tensor, time: tp.Optional[torch.Tensor] = None, **kwargs):
        """
        :param sample: MultiOrientedSample object
        :param freq: The frequency in Hz units
        :param time: It is used only for time resolved spectra
        :param kwargs:
        :return:
        """
        return super().__call__(sample, freq, time)

    def compute_parameters(self, sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           vector_down: torch.Tensor, vector_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           res_freq: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor]) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tp.Any]]:
        """
        :param sample: The sample which transitions must be found
        :param F: Magnetic free part of spin Hamiltonian H = F + B * G
        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param resonance_energies:
            Energies of spin states. The shape is [..., N]

        :param res_freq: Resonance frequencies. The shape os [..., N]

        :param full_system_vectors: Eigen vector of each level of a spin system. The shape os [..., N, N]

        :return: tuple of the next data
         - Resonance fields
         - Intensities of transitions
         - Width of transition lines
         - extras parameters computed in _compute_additional
        """

        intensities = self.intensity_calculator.compute_intensity(
            Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies, res_freq, full_system_vectors
        )
        lines_dimension = tuple(range(intensities.ndim - 1))
        intensities_mask = (intensities / intensities.abs().max() > self.threshold).any(dim=lines_dimension)
        intensities = intensities[..., intensities_mask]

        extras = self._mask_additional(vector_down,
            vector_up, lvl_down, lvl_up, resonance_energies,
            full_system_vectors)

        extras = self._mask_components(intensities_mask, *extras)

        res_fields = res_freq[..., intensities_mask]
        vector_u = vector_down[..., intensities_mask, :]
        vector_v = vector_up[..., intensities_mask, :]

        intensities = intensities / self.intensity_std
        width = self.broader(sample, vector_u, vector_v, res_fields)

        extras = self._compute_additional(
            sample, F, Gx, Gy, Gz, *extras
        )

        return res_fields, intensities, width, *extras

