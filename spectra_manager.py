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
                  compute_out: tuple,
                  gauss: torch.Tensor,
                  lorentz: torch.Tensor,
                  fields: torch.Tensor):
        _, res_fields, intensities, width, *additional_args = compute_out
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

    def _compute_field_straine_square(self, strained_data, vector_down, vector_up, B_trans, indexes):
        tensor_components, transformation_matrix = strained_data

        return (B_trans * (
                self._compute_element_field_dep(vector_up, tensor_components, transformation_matrix[indexes]) -
                self._compute_element_field_dep(vector_down, tensor_components, transformation_matrix[indexes])
                )).square()

    def _compute_field_free_straine_square(self, strained_data, vector_down, vector_up, indexes):
        tensor_components_A, tensor_components_B, transformation_matrix = strained_data
        return (
                self._compute_element_field_free(
                    vector_up, tensor_components_A, tensor_components_B, transformation_matrix[indexes]
                ) -
                self._compute_element_field_free(
                    vector_down, tensor_components_A, tensor_components_B, transformation_matrix[indexes]
                )
        ).square()

    def __call__(self, sample, vector_down, vector_up, B_trans, indexes):
        target_shape = vector_down.shape[:-1]
        device = B_trans.device
        dtype = B_trans.dtype

        result = torch.zeros(target_shape, dtype=dtype, device=device)

        for strained_data in sample.build_field_dep_straine():
            result += self._compute_field_straine_square(strained_data, vector_down, vector_up, B_trans, indexes)

        for strained_data in sample.build_zero_field_straine():
            result += self._compute_field_free_straine_square(strained_data, vector_down, vector_up, indexes)

        return result

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

    def _compute_magnitization(self, Gx, Gy, vector_down, vector_up, indexes):
        magnitization = compute_matrix_element(vector_down, vector_up, Gx[indexes]).square().abs() + \
                        compute_matrix_element(vector_down, vector_up, Gy[indexes]).square().abs()
        return magnitization

    def compute_intensity(self, Gx, Gy, Gz, batch):
        raise NotImplementedError

    def __call__(self, Gx, Gy, Gz, batch, *args, **kwargs):
        return self.compute_intensity(Gx, Gy, Gz, batch)


class StationaryIntensitiesCalculator(BaseIntensityCalculator):
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float] = None,
                 populator: tp.Optional[tp.Callable] = None, tolerancy: float = 1e-14):
        super().__init__(spin_system_dim, temperature, populator, tolerancy)

    def _init_populator(self, populator, temperature):
        if populator is None:
            return StationaryPopulator(temperature)
        else:
            return populator


    def compute_intensity(self, Gx, Gy, Gz, batch):
        """Base method to compute intensity (to be overridden)."""
        (vector_down, vector_up), (lvl_down, lvl_up), \
            B_trans, mask_triu, indexes, resonance_energies, _ = batch
        intensity = self.populator(resonance_energies, lvl_down, lvl_up) * (
                self._compute_magnitization(Gx, Gy, vector_down, vector_up, indexes)
        )
        return intensity


class TimeResolvedIntensitiesCalculator(BaseIntensityCalculator):
    def __init__(self, spin_system_dim: int, temperature: tp.Optional[float],
                 populator: tp.Optional[BaseTimeDependantPopulator],
                 tolerancy=1e-14):
        super().__init__(spin_system_dim, temperature, populator, tolerancy)

    def compute_intensity(self, Gx, Gy, Gz, batch):
        (vector_down, vector_up), (_, _),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies, _ = batch

        intensity = mask_trans * (
                self._compute_magnitization(Gx, Gy, vector_down, vector_up, indexes)
        )
        return intensity

    def calculate_population_evolution(self, time: torch.Tensor,
                                       res_fields, mask_triu,
                                       resonance_energies, vector_down, vector_up,
                                       *args, **kwargs):
        lvl_down, lvl_up = torch.triu_indices(self.spin_system_dim,
                                              self.spin_system_dim, offset=1)
        lvl_down = lvl_down[mask_triu]
        lvl_up = lvl_up[mask_triu]
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

        self.threshold = torch.tensor(1e-5)
        self.spin_system_dim, self.batch_dims, self.mesh =\
            self._init_sample_parameters(sample, spin_system_dim, batch_dims, mesh)
        self.mesh_size = self.mesh.initial_size
        self.broader = Broadener()

        self.res_field = res_field_algorithm.ResField(spin_dim = self.spin_system_dim,
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

    def _freq_to_field(self, vector_down, vector_up, Gz, indexes):
        """Compute frequency-to-field contribution"""
        factor_1 = compute_matrix_element(vector_up, vector_up, Gz[indexes])
        factor_2 = compute_matrix_element(vector_down, vector_down, Gz[indexes])

        diff = (factor_1 - factor_2).abs()
        safe_diff = torch.where(diff < self.tolerancy, self.tolerancy, diff)
        return safe_diff.reciprocal()

    def _get_output_eigenvector(self) -> bool:
        return False

    def _get_param_specs(self) -> list[ParamSpec]:
        return []

    def __call__(self,
                 sample: spin_system.MultiOrientedSample,
                 resonance_frequency: torch.Tensor,
                 fields: torch.Tensor, **kwargs):
        B_low  = fields[..., 0].unsqueeze(-1).expand(*self.mesh_size)
        B_high = fields[..., -1].unsqueeze(-1).expand(*self.mesh_size)

        F, Gx, Gy, Gz = sample.get_hamiltonian_terms()


        batches = self.res_field(sample, resonance_frequency, B_low, B_high, F, Gz)
        compute_out = self.compute_parameters(sample, F, Gx, Gy, Gz, batches)

        compute_out, fields = self._postcompute_batch_data(compute_out, fields, F, Gx, Gy, Gz, **kwargs)

        gauss = sample.gauss
        lorentz = sample.lorentz
        return self._finalize(compute_out, gauss, lorentz, fields)

    def _postcompute_batch_data(self, compute_out: tuple, fields: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor, **kwargs):
        return compute_out, fields

    def _finalize(self,
                  compute_out: tuple,
                  gauss: torch.Tensor,
                  lorentz: torch.Tensor,
                  fields: torch.Tensor):
        return self.spectra_processor.finalize(compute_out, gauss, lorentz, fields)

    def _precompute_batch_data(self, sample: spin_system.MultiOrientedSample, F, Gx, Gy, Gz, batch):
        intensity = self.intensity_calculator(Gx, Gy, Gz, batch)
        (vector_down, vector_up), (_, _),\
            B_trans, mask_triu, indexes, resonance_energies, _ = batch

        freq_to_field_val = self._freq_to_field(vector_down, vector_up, Gz, indexes)

        width_square = self.broader(sample, vector_down, vector_up, B_trans, indexes)
        return mask_triu, B_trans, intensity, width_square, indexes, freq_to_field_val

    def _filter_by_max(self, occurrences_max, global_max):
        updated_occurrences = []
        for bi, row_idx, col_idx, pair_max in occurrences_max:
            keep_local = torch.nonzero(pair_max / global_max >= self.threshold, as_tuple=False).squeeze(-1)
            new_cold_idx = col_idx[keep_local]
            if new_cold_idx.numel() > 0:
                updated_occurrences.append((bi, row_idx, new_cold_idx, keep_local))
        return updated_occurrences

    def _assign_global_indexes(self,
            occurrences: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[
        list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        int
    ]:
        """
        'occurrences' is list of the next data:
        the batch index
        The row-batch indexes
        The column indexes with respect to the triangular matrix of transitions
        The keep_local. - The pairs that were saved in filter by max
        The algorithm create global indexes

        Example 1
        row = [1 ,2, 3], col = [1, 2]
        row = [1, 2, 3], col = [3, 4]
        the output
        row = [1 , 2, 3], col = [1, 2], glob_idx = [0, 1]
        row = [1 , 2, 3], col = [3,  4], glob_idx = [2, 3]

        Example 2
        row = [1 ,2, 3], col = [1, 2]
        row = [4, 5, 6], col = [1, 2]
        the output
        row = [1 , 2, 3], col = [1, 2], glob_idx = [0, 1]
        row = [4 , 5, 6], col = [1,  2], glob_idx = [0, 1]

        Example 3
        row = [1 ,2], col = [1, 2]
        row = [2, 3], col = [2, 3]
        the output
        row = [1 , 2], col = [1, 2], glob_idx = [0, 1]
        row = [2 , 3], col = [2,  3], glob_idx = [2, 3]

        Example 4
        row = [1 ,2 , 3], col = [1, 2]
        row = [4, 5, 6], col = [2, 3]
        the output
        row = [1 , 2, 3], col = [1, 2], glob_idx = [0, 1]
        row = [4 , 5, 6], col = [2,  3], glob_idx = [1, 2]
        """
        col_to_slots = {}

        out: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        max_global = -1

        for (batch_idx, rows, cols, keep_pairs) in occurrences:
            rows_set = set(rows.tolist())
            assigned_indices: list[int] = []
            for c in cols.tolist():
                slots = col_to_slots.get(int(c), [])
                reused = False
                for slot_idx, (gidx, used_rows) in enumerate(slots):
                    if used_rows.isdisjoint(rows_set):
                        assigned_indices.append(gidx)
                        slots[slot_idx] = (gidx, used_rows.union(rows_set))
                        reused = True
                        break

                if not reused:
                    max_global += 1
                    gidx = max_global
                    assigned_indices.append(gidx)
                    slots.append((gidx, set(rows_set)))
                    col_to_slots[int(c)] = slots

            global_idx_tensor = torch.tensor(assigned_indices, dtype=torch.long)
            out.append((batch_idx, rows, cols, global_idx_tensor, keep_pairs))

        return out, max_global + 1

    def _first_pass(self, sample, F, Gx, Gy, Gz, batches):
        cached_batches = []
        occurences_max = []

        global_max = torch.tensor(0)
        for bi, batch in enumerate(batches):
            mask_triu, B_trans_batch, intensity_batch, width_square_batch, mask_indexes,\
                freq_to_field_val, *extras = \
                self._precompute_batch_data(sample, F, Gx, Gy, Gz, batch)

            row_idx = torch.nonzero(mask_indexes, as_tuple=False).squeeze(-1)
            col_idx = torch.nonzero(mask_triu, as_tuple=False).squeeze(-1)


            if row_idx.numel() > 0 and col_idx.numel() > 0:

                pair_max = torch.amax(torch.abs(intensity_batch), dim=tuple(range(intensity_batch.ndim - 1)))
                global_max = torch.max(global_max, torch.max(pair_max))
                cached_batches.append(
                    (mask_triu, B_trans_batch,
                     intensity_batch,
                     width_square_batch,
                     mask_indexes, freq_to_field_val, *extras)
                )
                occurences_max.append((bi, row_idx, col_idx, pair_max))

        return cached_batches, self._filter_by_max(occurences_max, global_max)

    def _match_col_indexes(self, kept_pairs: torch.Tensor, col_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param kept_pairs: The indexes that must be saved because the intensity
        for some transitions is quite large. It is 1-dim tensor.
        :param col_idx: The indexes of allowed transitions. It is 1-dim tensor.
        :return: idx1, idx2. Where idx1 is mask over kept_pairs that appear in col_idx,
        idx2 is mask over col_idx that appear in kept_pairs,
        """
        idx1 = torch.nonzero(torch.isin(kept_pairs, col_idx), as_tuple=True)[0]
        idx2 = torch.nonzero(torch.isin(col_idx, kept_pairs), as_tuple=True)[0]
        return idx1, idx2

    def _init_extras(self, config_dims, m: int, device: torch.device):
        """
        Allocate all full-sized storage tensors based on param_specs.
        """
        extra_tensors = []
        for param_spec in self._param_specs:
            if param_spec.category == "scalar":
                new_tensor = torch.zeros((*config_dims, m), dtype=param_spec.dtype,
                                         device=device)
            elif param_spec.category == "vector":
                new_tensor = torch.zeros((*config_dims, m, self.spin_system_dim), dtype=param_spec.dtype,
                                         device=device)
            elif param_spec.category == "matrix":
                new_tensor = torch.zeros((*config_dims, m, self.spin_system_dim, self.spin_system_dim),
                                         dtype=param_spec.dtype, device=device)
            else:
                raise ValueError("Wrong category")
            extra_tensors.append(new_tensor)
        return extra_tensors

    def _update_extras(self, extra_tensors: list[torch.Tensor], extras_batch: list[torch.Tensor],
                       row_idx: torch.Tensor, col_base_idx: torch.Tensor, col_batch_idx: torch.Tensor):

        for idx, param_spec in enumerate(self._param_specs):
            if param_spec.category == "scalar":
                extra_tensors[idx][row_idx[:, None], col_base_idx] = \
                    extras_batch[idx][..., col_batch_idx]

            elif param_spec.category == "vector":
                extra_tensors[idx][row_idx[:, None], col_base_idx, :] = \
                    extras_batch[idx][..., col_batch_idx, :]

            elif param_spec.category == "matrix":
                extra_tensors[idx][row_idx[:, None], col_base_idx, :, :] = \
                    extras_batch[idx][..., col_batch_idx, :, :]

    def compute_parameters(self, sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           batches: list[
                               tuple[
                                    tuple[torch.Tensor, torch.Tensor],
                                    tuple[torch.Tensor, torch.Tensor],
                                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor | None,
                               ]]):
        """
        :param sample: The sample which transitions must be found
        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :param batches: list of next data:
        - tuple of eigen vectors for resonance energy levels of low and high levels
        - tuple of valid indexes of levels between which transition occurs
        - magnetic field of transitions
        - mask_trans - Boolean transition mask [batch_size, num valid transitions]. It is True if transition occurs
        - mask_triu - Boolean triangular mask [total number of all possible transitions].
        It is True if there is at least one element in batch for which transition between energy levels occurs.
        num valid transitions = sum(mask_triu)
        - indexes: Boolean mask of correct elements in batch
        - resonance energies
        - vector_full_system | None. The eigen vectors for all energy levels

        :return:
        """

        config_dims = (*self.batch_dims, *self.mesh_size)
        batches, occurrences = self._first_pass(sample, F, Gx, Gy, Gz, batches)
        occurrences, max_columns = self._assign_global_indexes(occurrences)

        res_fields = torch.zeros((*config_dims, max_columns), dtype=torch.float32, device=Gx.device)
        intensities = torch.zeros((*config_dims, max_columns), dtype=torch.float32, device=Gx.device)
        width_square = torch.zeros((*config_dims, max_columns), dtype=torch.float32, device=Gx.device)
        freq_to_field_global = torch.zeros((*config_dims, max_columns), dtype=torch.float32, device=Gx.device)

        extra_tensors = self._init_extras(config_dims, max_columns, Gx.device)
        num_pairs = (self.spin_system_dim ** 2 - self.spin_system_dim) // 2
        mask_triu_general = torch.zeros(num_pairs, dtype=torch.bool)

        for bi, rows_batch_idx, col_batch_idx, col_base_idx, keep_local in occurrences:
            mask_triu, B_trans_batch, intensity_batch,\
                width_square_batch, mask_indexes, freq_to_field_val, *extras_batch = batches[bi]
            row_idx = torch.nonzero(mask_indexes).squeeze(-1)
            if row_idx.numel() > 0 and col_base_idx.numel() > 0:
                # mask_triu_general[..., col_batch_idx] = True
                res_fields[row_idx[:, None], col_base_idx] = B_trans_batch[..., keep_local]

                intensities[row_idx[:, None], col_base_idx] += intensity_batch[..., keep_local]
                width_square[row_idx[:, None], col_base_idx] += width_square_batch[..., keep_local]

                freq_to_field_global[row_idx[:, None], col_base_idx] += freq_to_field_val[..., keep_local]

                self._update_extras(extra_tensors, extras_batch, row_idx, col_base_idx, keep_local)

        intensities *= freq_to_field_global
        intensities = intensities / intensities.abs().max()
        width = self.broader.add_hamiltonian_straine(sample, width_square) * freq_to_field_global

        return mask_triu_general, res_fields, intensities, width, *extra_tensors


class StationarySpectraCreator(BaseSpectraCreator):
    def _postcompute_batch_data(self, compute_out: tuple, fields: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor, Gz: torch.Tensor):
        return compute_out, fields

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
            ParamSpec("vector", torch.float32),
            ParamSpec("vector", torch.complex64),
            ParamSpec("vector", torch.complex64)
            ]
        return params

    # REWRITE IT TO MAKE LOGIC BETTER
    def _precompute_batch_data(self, sample: spin_system.MultiOrientedSample, F, Gx, Gy, Gz, batch):
        """
        :param sample:
        :param Gx:
        :param Gy:
        :param Gz:
        :param batch:
        :return:
        """
        intensity = self.intensity_calculator(Gx, Gy, Gz, batch)
        (vector_down, vector_up), (lvl_down, lvl_up),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies, _ = batch
        width_square = self.broader(sample, vector_down, vector_up, B_trans, indexes)

        freq_to_field_val = self._freq_to_field(vector_down, vector_up, Gz, indexes)
        return mask_trans, mask_triu, B_trans,\
            intensity, width_square, indexes, freq_to_field_val, resonance_energies, vector_down, vector_up

    def _postcompute_batch_data(self, compute_out: tuple, fields: torch.Tensor,
                                F: torch.Tensor, Gx: torch.Tensor, Gy: torch.Tensor,
                                Gz: torch.Tensor, time: torch.Tensor):
        mask_triu, res_fields, intensities, width, *extras = compute_out
        population = self.intensity_calculator.calculate_population_evolution(
            time, res_fields, mask_triu, *extras
        )
        intensities = (intensities.unsqueeze(0) * population)
        return (mask_triu, res_fields, intensities, width), fields

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
            ParamSpec("vector", torch.float32),
            ParamSpec("vector", torch.complex64),
            ParamSpec("vector", torch.complex64),
            ParamSpec("matrix", torch.complex64)
            ]
        return params

    def _precompute_batch_data(self, sample: spin_system.MultiOrientedSample, F, Gx, Gy, Gz, batch):
        """
        :param sample:
        :param Gx:
        :param Gy:
        :param Gz:
        :param batch:
        :return:
        """
        intensity = self.intensity_calculator(Gx, Gy, Gz, batch)
        (vector_down, vector_up), (lvl_down, lvl_up),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies, vector_full = batch
        width_square = self.broader(sample, vector_down, vector_up, B_trans, indexes)

        freq_to_field_val = self._freq_to_field(vector_down, vector_up, Gz, indexes)
        return mask_trans, mask_triu, B_trans, intensity, width_square, indexes, freq_to_field_val, resonance_energies,\
            vector_down, vector_up, vector_full


class MultiSampleCreator:
    def __init__(self,
                 creators: list[BaseSpectraCreator],
                 temperature: tp.Optional[float] = 293,
                 intensity_calculator: tp.Optional[MultiSampleIntensitiesCalculator] = None,
                 populator: tp.Optional[BaseTimeDependantPopulator] = None,
                 weights: tp.Optional[torch.Tensor] = None,
                 ):
        """
        creators[i] is already configured for sample i (its spin_system_dim, mesh, …).
        """
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

