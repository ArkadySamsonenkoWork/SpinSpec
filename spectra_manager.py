import warnings
import typing as tp
from dataclasses import dataclass
from abc import ABC, abstractmethod

from spectral_integration import BaseSpectraIntegrator,\
    SpectraIntegratorEasySpinLike,\
    SpectraIntegratorEasySpinLikeTimeResolved, MeanIntegrator
from population import BaseTimeDependantPopulator, StationaryPopulator
from population.mechanisms import T1Population

import torch
from torch import nn
import torch.fft as fft

import constants
import mesher
import res_field_algorithm
import spin_system

from typing import List, Tuple, Set, Dict
from collections import defaultdict


def compute_matrix_element(vector_down, vector_up, G):
    tmp = torch.matmul(G, vector_up.transpose(-2, -1))  # (..., i, b)
    tmp = tmp.transpose(-2, -1)  # (..., b, i)
    return (vector_down.conj() * tmp).sum(dim=-1)


class PostSpectraProcessing:
    def __init__(self, *args, **kwargs):
        """
        :param gauss: The gauss parameter. The shape is [batch_size] or []
        :param lorentz: The gauss parameter. The shape is [batch_size] or []
        """
        pass

    def _skip_broader(self, gauss, lorentz, magnetic_fields: torch.Tensor, spec: torch.Tensor):
        return spec

    def _broading_fabric(self, gauss: torch.Tensor, lorentz: torch.Tensor):
        if (gauss == 0) and (lorentz == 0):
            return self._skip_broader

        elif (gauss != 0) and (lorentz == 0):
            return self._gauss_broader

        elif (gauss == 0) and (lorentz != 0):
            return self._lorentz_broader

        else:
            return self._voigt_broader

    def __call__(self, gauss: torch.Tensor, lorentz: torch.Tensor,
                 magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param magnetic_field: Tensor of shape [batch, N]
        :param spec: Spectrum tensor of shape [batch, ..., N]
        :return: Broadended spectrum, same shape as spec
        """
        _broading_method = self._broading_fabric(gauss, lorentz)
        return _broading_method(gauss, lorentz, magnetic_field, spec)

    def _build_lorentz_kernel(self, magnetic_field: torch.Tensor, fwhm_lorentz: torch.Tensor):
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        N = magnetic_field.shape[-1]
        idx = torch.arange(N, device=magnetic_field.device) - N//2
        x = idx * dH
        gamma = fwhm_lorentz / 2
        L = (gamma / torch.pi) / (x ** 2 + gamma ** 2)
        return L

    def _build_gauss_kernel(self, magnetic_field: torch.Tensor, fwhm_gauss: torch.Tensor):
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        N = magnetic_field.shape[-1]
        idx = torch.arange(N, device=magnetic_field.device) - N//2
        x = idx * dH
        sigma = fwhm_gauss / (2 * (2 * torch.log(torch.tensor(2.0)))**0.5)
        G = torch.exp(-0.5 * (x / sigma)**2) / (sigma * (2 * torch.pi)**0.5)
        return G

    def _build_voigt_kernel(self,
                            magnetic_field: torch.Tensor,
                            fwhm_gauss: torch.Tensor,
                            fwhm_lorentz: torch.Tensor
                            ):
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
        """Apply convolution via FFT per batch."""
        S = fft.rfft(spec, dim=-1)
        K = fft.rfft(torch.fft.ifftshift(kernel, dim=-1), dim=-1)
        out = fft.irfft(S * K, n=spec.shape[-1], dim=-1)
        return out

    def _gauss_broader(self,
                       gauss: torch.Tensor, lorentz: torch.Tensor,
                       magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        batch = magnetic_field.shape[0]
        fwhm_gauss = gauss.expand(batch)
        kernel = self._build_gauss_kernel(magnetic_field, fwhm_gauss)
        return self._apply_convolution(spec, kernel)

    def _lorentz_broader(self,
                         gauss: torch.Tensor, lorentz: torch.Tensor,
                         magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        batch = magnetic_field.shape[0]
        fwhm_lorentz = lorentz.expand(batch)
        kernel = self._build_lorentz_kernel(magnetic_field, fwhm_lorentz)
        return self._apply_convolution(spec, kernel)

    def _voigt_broader(self, gauss: torch.Tensor, lorentz: torch.Tensor,
                       magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        batch = magnetic_field.shape[0]
        fwhm_gauss = gauss.expand(batch)
        fwhm_lorentz = lorentz.expand(batch)
        kernel = self._build_voigt_kernel(magnetic_field, fwhm_gauss, fwhm_lorentz)
        return self._apply_convolution(spec, kernel)


class IntegrationProcessorBase(ABC):
    def __init__(self,
                 mesh: mesher.BaseMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing()):
        self.threshold = torch.tensor(1e-5)
        self.spectra_integrator = self._init_spectra_integrator(spectra_integrator, harmonic)
        self.mesh = mesh
        self.post_spectra_processor = post_spectra_processor

    @abstractmethod
    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int):
        pass

    @abstractmethod
    def _compute_areas(self, expanded_size: torch.Tensor):
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

    def finalize(self,
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
        spec = self.spectra_integrator.integrate(
            res_fields, width, intensities, areas, fields
        )
        return self.post_spectra_processor(gauss, lorentz, fields, spec)


class IntegrationProcessorPowder(IntegrationProcessorBase):
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing()):
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor)

    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int):
        if spectra_integrator is None:
            return SpectraIntegratorEasySpinLike(harmonic)
        else:
            return spectra_integrator

    def _compute_areas(self, expanded_size: torch.Tensor):
        grid, simplices = self.mesh.post_mesh
        areas = self.mesh.spherical_triangle_areas(grid, simplices)
        areas = areas.reshape(1, -1).expand(expanded_size, -1).flatten()
        return areas

    def _process_tensor(self, data_tensor: torch.Tensor):
        _, simplices = self.mesh.post_mesh
        processed = self.mesh.post_process(data_tensor.transpose(-1, -2))
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
        :param intensities: the tensor of resonance fields. The shape is [..., num_resonance fields]
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
        areas = self._compute_areas(expanded_size)
        return res_fields, width, intensities, areas


class IntegrationProcessorCrystal(IntegrationProcessorBase):
    def __init__(self,
                 mesh: mesher.CrystalMesh,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing()):
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor)

    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int):
        if spectra_integrator is None:
            return MeanIntegrator(harmonic)
        else:
            return spectra_integrator

    def _compute_areas(self, expanded_size: torch.Size):
        areas = torch.ones(expanded_size, dtype=torch.float32)
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
        areas = self._compute_areas(expanded_size)
        return res_fields, width, intensities, areas


class IntegrationProcessorTimeResolved(IntegrationProcessorPowder):
    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int):
        if spectra_integrator is None:
            return SpectraIntegratorEasySpinLikeTimeResolved(harmonic)
        else:
            return spectra_integrator

    def _compute_areas(self, expanded_size: torch.Tensor):
        grid, simplices = self.mesh.post_mesh
        areas = self.mesh.spherical_triangle_areas(grid, simplices)
        areas = areas.reshape(1, -1).expand(expanded_size, -1).flatten()
        return areas

    def _transform_data_to_mesh_format_format(self, res_fields, intensities, width):
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
        areas = self._compute_areas(expanded_size)
        return res_fields, width, intensities, areas


class Broadener:
    def __init__(self):
        pass

    def _compute_element_field_free(self, vector: torch.Tensor,
                          tensor_components_A: torch.Tensor, tensor_components_B: torch.Tensor,
                          transformation_matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            '...pij,jkl,ikl,...bk,...bl->...b',
            transformation_matrix, tensor_components_A, tensor_components_B, torch.conj(vector), vector
        ).real

    def _compute_element_field_dep(self, vector: torch.Tensor,
                          tensor_components: torch.Tensor,
                          transformation_matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            '...pi, ikl,...bk,...bl->...b',
            transformation_matrix, tensor_components, torch.conj(vector), vector
        ).real

    def _compute_field_straine_square(self, strained_data, vector_down, vector_up, B_trans):
        tensor_components, transformation_matrix = strained_data

        return (B_trans * (
                self._compute_element_field_dep(vector_up, tensor_components, transformation_matrix) -
                self._compute_element_field_dep(vector_down, tensor_components, transformation_matrix)
                )).square()

    def _compute_field_free_straine_square(self, strained_data, vector_down, vector_up):
        tensor_components_A, tensor_components_B, transformation_matrix = strained_data
        return (
                self._compute_element_field_free(
                    vector_up, tensor_components_A, tensor_components_B, transformation_matrix
                ) -
                self._compute_element_field_free(
                    vector_down, tensor_components_A, tensor_components_B, transformation_matrix
                )
        ).square()

    def __call__(self, sample, vector_down, vector_up, B_trans):
        target_shape = vector_down.shape[:-1]
        device = B_trans.device
        dtype = B_trans.dtype

        result = torch.zeros(target_shape, dtype=dtype, device=device)

        for strained_data in sample.build_field_dep_straine():
            result += self._compute_field_straine_square(strained_data, vector_down, vector_up, B_trans)

        for strained_data in sample.build_zero_field_straine():
            result += self._compute_field_free_straine_square(strained_data, vector_down, vector_up)

        return self.add_hamiltonian_straine(sample, result)

    def add_hamiltonian_straine(self, sample: spin_system.MultiOrientedSample, squared_width):
        hamiltonian_width = sample.build_hamiltonian_straineed().unsqueeze(-1).square()
        return (squared_width + hamiltonian_width).sqrt()


class BaseIntensityCalculator:
    def __init__(self, spin_system_dim: int | list[int],
                 temperature: tp.Optional[float] = None,
                 populator: tp.Optional[tp.Callable] = None,
                 tolerancy: float = 1e-12):
        self.tolerancy = torch.tensor(tolerancy)
        self.populator = self._init_populator(populator, temperature)
        self.spin_system_dim = spin_system_dim
        self.temperature = temperature

    def _init_populator(self, populator: tp.Optional[tp.Callable], temperature: tp.Optional[float]):
        return populator

    def _compute_magnitization(self, Gx, Gy, vector_down, vector_up):
        magnitization = compute_matrix_element(vector_down, vector_up, Gx).square().abs() + \
                        compute_matrix_element(vector_down, vector_up, Gy).square().abs()
        return magnitization

    def compute_intensity(self, Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies):
        raise NotImplementedError

    def __call__(self, Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies, *args, **kwargs):
        return self.compute_intensity(Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies)


class StationaryIntensitiesCalculator(BaseIntensityCalculator):
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float] = None,
                 populator: tp.Optional[tp.Callable] = None, tolerancy: float = 1e-14):
        super().__init__(spin_system_dim, temperature, populator, tolerancy)

    def _init_populator(self, populator, temperature):
        if populator is None:
            return StationaryPopulator(temperature)
        else:
            return populator

    def compute_intensity(self, Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies):
        """Base method to compute intensity (to be overridden)."""
        intensity = self.populator(resonance_energies, lvl_down, lvl_up) * (
                self._compute_magnitization(Gx, Gy, vector_down, vector_up)
        )
        return intensity


class TimeResolvedIntensitiesCalculator(BaseIntensityCalculator):
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float],
                 populator: tp.Optional[BaseTimeDependantPopulator],
                 tolerancy=1e-14):
        super().__init__(spin_system_dim, temperature, populator, tolerancy)

    def compute_intensity(self,Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies):
        intensity = (
                self._compute_magnitization(Gx, Gy, vector_down, vector_up)
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
                 tolerancy=1e-12
                 ):
        super().__init__(spin_system_dim, temperature, populator, tolerancy)

    def calculate_population_evolution(self, time: torch.Tensor, intensity_outs, spin_dimensions: list[int]):
        populations = self.populator(time, intensity_outs, spin_dimensions)
        return populations


@dataclass
class ParamSpec:
    category: str
    dtype: torch.dtype

    def __post_init__(self):
        assert self.category in (
            "scalar", "vector", "matrix"), f"Category must be one of 'scalar', 'vector', 'matrix', got {self.category}"


class BaseSpectraCreator(ABC):
    def __init__(self,
                 freq: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[float] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[tp.Callable] = None,
                 populator: tp.Optional[StationaryPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[float] = 293,
                 ):

        self.resonance_freq = torch.tensor(freq)
        self.threshold = torch.tensor(1e-5)
        self.spin_system_dim, self.batch_dims, self.mesh =\
            self._init_sample_parameters(sample, spin_system_dim, batch_dims, mesh)
        self.mesh_size = self.mesh.initial_size
        self.broader = Broadener()

        self.res_field = res_field_algorithm.ResField(spin_system_dim=self.spin_system_dim,
                                                      mesh_size=self.mesh_size,
                                                      batch_dims=self.batch_dims,
                                                      output_full_eigenvector=self._get_output_eigenvector())

        self.intensity_calculator = self._get_intenisty_calculator(intensity_calculator, temperature, populator)
        self._param_specs = self._get_param_specs()

        self.tolerancy = torch.tensor(1e-10)
        self._vacuum_g_factor = torch.tensor(2.002)
        self.spectra_processor = self._init_spectra_processor(spectra_integrator,
                                                              harmonic,
                                                              post_spectra_processor)

        if populator is not None:
            self.temperature = None

    @abstractmethod
    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing) -> IntegrationProcessorBase:
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
            spin_system_dim = sample.spin_system.dim
            batch_dims = sample.config_shape[:-1]
            mesh = sample.mesh

        return spin_system_dim, batch_dims, mesh

    def _get_intenisty_calculator(self, intensity_calculator, temperature: float, populator: StationaryPopulator):
        if intensity_calculator is None:
            return StationaryIntensitiesCalculator(self.spin_system_dim, temperature, populator)
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
        return []

    def __call__(self,
                 sample: spin_system.MultiOrientedSample,
                 fields: torch.Tensor, **kwargs):
        B_low  = fields[..., 0].unsqueeze(-1).expand(*self.mesh_size)
        B_high = fields[..., -1].unsqueeze(-1).expand(*self.mesh_size)

        F, Gx, Gy, Gz = sample.get_hamiltonian_terms()

        (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up), res_fields, resonance_energies, full_eigen_vectors =\
            self.res_field(sample, self.resonance_freq, B_low, B_high, F, Gz)

        res_fields, intensities, width, *extras = self.compute_parameters(sample, F, Gx, Gy, Gz,
                                              vectors_u, vectors_v,
                                              valid_lvl_down, valid_lvl_up,
                                              res_fields,
                                              resonance_energies,
                                              full_eigen_vectors
                                              )

        res_fields, intensities, width = self._postcompute_batch_data(
            res_fields, intensities, width, F, Gx, Gy, Gz, *extras, **kwargs
        )

        gauss = sample.gauss
        lorentz = sample.lorentz

        return self._finalize(res_fields, intensities, width, gauss, lorentz, fields)

    def _postcompute_batch_data(self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor, *extras,  **kwargs):
        return res_fields, intensities, width

    def _finalize(self,
                  res_fields: torch.Tensor,
                  intensities: torch.Tensor,
                  width: torch.Tensor,
                  gauss: torch.Tensor,
                  lorentz: torch.Tensor,
                  fields: torch.Tensor):
        return self.spectra_processor.finalize(res_fields, intensities, width, gauss, lorentz, fields)

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


    def _mask_additional(self, vectors_down: torch.Tensor, vectors_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor], intensities_mask: torch.Tensor):
        return ()


    def _compute_additional(self,
                           sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                            *extras):
        return extras


    def compute_parameters(self, sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           vectors_down: torch.Tensor, vectors_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           res_fields: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor]):
        """
        :param sample: The sample which transitions must be found
        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :return:
        """

        intensities = self.intensity_calculator.compute_intensity(
            Gx, Gy, Gz, vectors_down, vectors_up, lvl_down, lvl_up, resonance_energies
        )

        intensities_mask = (intensities / intensities.abs().max() > self.threshold).any(dim=-2)
        intensities = intensities[..., intensities_mask]

        extras = self._mask_additional(vectors_down,
            vectors_up, lvl_down, lvl_up, resonance_energies,
            full_system_vectors, intensities_mask)


        res_fields = res_fields[..., intensities_mask]
        vectors_u = vectors_down[..., intensities_mask, :]
        vectors_v = vectors_up[..., intensities_mask, :]

        freq_to_field = self._freq_to_field(vectors_u, vectors_v, Gz)
        intensities *= freq_to_field
        intensities = intensities / intensities.abs().max()
        width = self.broader(sample, vectors_u, vectors_v, res_fields) * freq_to_field

        extras = self._compute_additional(
            sample, F, Gx, Gy, Gz, *extras
        )

        return res_fields, intensities, width, *extras


class StationarySpectraCreator(BaseSpectraCreator):
    def _postcompute_batch_data(self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor):
        return res_fields, intensities, width

    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing) -> IntegrationProcessorBase:
        if self.mesh.name == "PowderMesh":
            return IntegrationProcessorPowder(self.mesh, spectra_integrator, harmonic, post_spectra_processor)

        elif self.mesh.name == "CrystalMesh":
            return IntegrationProcessorCrystal(self.mesh, spectra_integrator, harmonic, post_spectra_processor)

        else:
            return IntegrationProcessorPowder(self.mesh, spectra_integrator, harmonic, post_spectra_processor)


class TruncatedSpectraCreatorTimeResolved(BaseSpectraCreator):
    def _init_spectra_integrator(self, spectra_integrator: tp.Optional[BaseSpectraIntegrator], harmonic: int):
        if spectra_integrator is None:
            self.spectra_integrator = SpectraIntegratorEasySpinLikeTimeResolved(harmonic)
        else:
            self.spectra_integrator = spectra_integrator

    def _get_intenisty_calculator(self, intensity_calculator,
                                  temperature,
                                  populator: tp.Optional[BaseTimeDependantPopulator]):
        if intensity_calculator is None:
            return TimeResolvedIntensitiesCalculator(self.spin_system_dim, temperature, populator)
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

    def _mask_additional(self, vectors_down: torch.Tensor, vectors_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor], intensities_mask: torch.Tensor):

        return lvl_down, lvl_up, resonance_energies, vectors_down, vectors_up

    def _postcompute_batch_data(self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor, time: torch.Tensor, *extras):
        lvl_down, lvl_up, resonance_energies, vectors_down, vectors_up, *extras = extras

        population = self.intensity_calculator.calculate_population_evolution(
            time, res_fields, lvl_down, lvl_up, resonance_energies, vectors_down, vectors_up, *extras
        )
        intensities = (intensities.unsqueeze(0) * population)
        return res_fields, intensities, width

    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing) -> IntegrationProcessorBase:
        if self.mesh.name == "PowderMesh":
            return IntegrationProcessorTimeResolved(self.mesh, spectra_integrator, harmonic, post_spectra_processor)
        else:
            return IntegrationProcessorTimeResolved(self.mesh, spectra_integrator, harmonic, post_spectra_processor)


class CoupledSpectraCreatorTimeResolved(TruncatedSpectraCreatorTimeResolved):
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

    def _compute_additional(self, sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           vectors_down: torch.Tensor, vectors_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           res_fields: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor], *extras):
        """
        :param sample:
        :param Gx:
        :param Gy:
        :param Gz:
        :param batch:
        :return:
        """

        return lvl_down, lvl_up, resonance_energies, vectors_down, vectors_up, full_system_vectors


class MultiSampleCreator:
    def __init__(self,
                 resonance_freq: tp.Union[float, torch.Tensor],
                 creators: list[BaseSpectraCreator],
                 temperature: tp.Optional[float] = 293,
                 intensity_calculator: tp.Optional[MultiSampleIntensitiesCalculator] = None,
                 populator: tp.Optional[BaseTimeDependantPopulator] = None,
                 weights: tp.Optional[torch.Tensor] = None,
                 ):
        """
        creators[i] is already configured for sample i (its spin_system_dim, mesh, …).
        """
        self.resonance_freq = resonance_freq
        self.spin_system_dim = [creator.spin_system_dim for creator in creators]
        if len(creators) == 0:
            raise ValueError("Need at least one creator")
        self.creators = list(creators)
        self.intensity_calculator = self._get_intenisty_calculator(intensity_calculator, temperature, populator)
        if weights is None:
            self.weights = torch.ones(len(creators), dtype=torch.float32)
        else:
            self.weights = weights
        self.mesh_size = self.creators[0].mesh_size

    def _get_intenisty_calculator(self,
                                  intensity_calculator: tp.Optional[MultiSampleIntensitiesCalculator],
                                  temperature: float,
                                  populator: tp.Optional[BaseTimeDependantPopulator]):
        if intensity_calculator is None:
            return MultiSampleIntensitiesCalculator(self.spin_system_dim, temperature, populator)
        else:
            return intensity_calculator

    def _postcompute_intesities(self,
                                time: torch.Tensor,
                                intensities_samples: list[tuple],
                                population_samples: list[tuple]):
        intensities_samples_finile = []
        population_samples = self.intensity_calculator.calculate_population_evolution(time,
                                                                                      population_samples,
                                                                                      self.spin_system_dim)
        for intensities, population in zip(intensities_samples, population_samples):
            intensities_samples_finile.append(intensities.unsqueeze(0) * population)
        return intensities_samples_finile

    def __call__(self,
                 samples: tp.Sequence[spin_system.MultiOrientedSample],
                 resonance_frequency: torch.Tensor,
                 fields: torch.Tensor, time: torch.Tensor
                 ) -> torch.Tensor:
        if len(samples) != len(self.creators):
            raise ValueError(f"Expected {len(self.creators)} samples, got {len(samples)}")

        B_low = fields[..., 0].unsqueeze(-1).expand(*self.mesh_size)
        B_high = fields[..., -1].unsqueeze(-1).expand(*self.mesh_size)

        intensities_samples = []
        widths_samples = []
        population_samples = []
        mask_triu_samples = []

        for sample, creator in zip(samples, self.creators):
            F, Gx, Gy, Gz = sample.get_hamiltonian_terms()

            batches = creator.res_field(sample, resonance_frequency, B_low, B_high, F, Gz)
            compute_out = creator.compute_parameters(sample, F, Gx, Gy, Gz, batches)

            mask_triu, res_fields, intensities, width, *extras = compute_out

            lvl_down, lvl_up = torch.triu_indices(creator.spin_system_dim,
                                                  creator.spin_system_dim, offset=1)
            lvl_down = lvl_down[mask_triu]
            lvl_up = lvl_up[mask_triu]

            intensities_samples.append(intensities)
            widths_samples.append(width)

            population_samples.append((res_fields, lvl_down, lvl_up, *extras, F, Gz))
            mask_triu_samples.append(mask_triu)

        intensities_samples = self._postcompute_intesities(
            time, intensities_samples, population_samples
        )

        spectras = []
        for idx, creator in enumerate(self.creators):
            compute_out = (mask_triu_samples[idx], population_samples[idx][0],
                           intensities_samples[idx], widths_samples[idx], *population_samples[idx][2:])
            gauss = samples[idx].gauss
            lorentz = samples[idx].lorentz
            spectra = creator._finalize(compute_out, gauss, lorentz, fields)
            spectras.append(spectra)
        return torch.stack(spectras, dim=0)

