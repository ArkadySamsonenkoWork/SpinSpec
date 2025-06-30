import warnings
import typing as tp
from dataclasses import dataclass

from spectral_integration import BaseSpectraIntegrator,\
    SpectraIntegratorExtended, SpectraIntegratorEasySpinLike, SpectraIntegratorEasySpinLikeTimeResolved
from population import BaseTimeDependantPopulator, StationaryPopulator
from population.mechanisms import T1Population

import torch
from torch import nn

import constants
import mesher
import res_field_algorithm
import spin_system



class Broadening:
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
        strained_field = sum(
            self._compute_field_straine_square(strained_data, vector_down, vector_up, B_trans, indexes)
            for strained_data in sample.build_field_dep_straine()
        )
        strained_free = sum(
            self._compute_field_free_straine_square(strained_data, vector_down, vector_up, indexes)
            for strained_data in sample.build_zero_field_straine()
        )
        return strained_field + strained_free

    def add_hamiltonian_straine(self, sample: spin_system.MultiOrientedSample, squared_width):
        hamiltonian_width = sample.build_hamiltonian_straineed().unsqueeze(-1).square()
        return (squared_width + hamiltonian_width).sqrt()


class BaseIntensityCalculator:
    def __init__(self, spin_system_dim: int, populator: tp.Callable, tolerancy=1e-14):
        self.tolerancy = torch.tensor(tolerancy)
        self.populator = populator
        self.spin_system_dim = spin_system_dim

    def compute_matrix_element(self, vector_down, vector_up, G):
        return torch.einsum('...bi,...ij,...bj->...b', torch.conj(vector_down), G, vector_up)

    def _compute_magnitization(self, Gx, Gy, vector_down, vector_up, indexes):
        magnitization = self.compute_matrix_element(vector_down, vector_up, Gx[indexes]).square().abs() + \
                        self.compute_matrix_element(vector_down, vector_up, Gy[indexes]).square().abs()
        return magnitization

    def _freq_to_field(self, vector_down, vector_up, Gz, indexes):
        """Compute frequency-to-field contribution"""
        factor_1 = self.compute_matrix_element(vector_up, vector_up, Gz[indexes])
        factor_2 = self.compute_matrix_element(vector_down, vector_down, Gz[indexes])
        diff = (factor_1 - factor_2).abs()
        safe_diff = torch.where(diff < self.tolerancy, self.tolerancy, diff)
        return safe_diff.reciprocal()

    def compute_intensity(self, Gx, Gy, Gz, batch):
        raise NotImplementedError

    def __call__(self, Gx, Gy, Gz, batch):
        return self.compute_intensity(Gx, Gy, Gz, batch)


class StationaryIntensitiesCalculator(BaseIntensityCalculator):
    def __init__(self, spin_system_dim: int, populator: tp.Callable = StationaryPopulator(), tolerancy=1e-14):
        super().__init__(spin_system_dim, populator, tolerancy)

    def compute_intensity(self, Gx, Gy, Gz, batch):
        """Base method to compute intensity (to be overridden)."""
        (vector_down, vector_up), (lvl_down, lvl_up), \
            B_trans, mask_trans, mask_triu, indexes, resonance_energies, _ = batch

        intensity = mask_trans * self.populator(resonance_energies, lvl_down, lvl_up) * (
                self._compute_magnitization(Gx, Gy, vector_down, vector_up, indexes) +
                self._freq_to_field(vector_down, vector_up, Gz, indexes)
        )
        return intensity


class TimeResolvedIntensitiesCalculator(BaseIntensityCalculator):
    def __init__(self, spin_system_dim: int, time: torch.Tensor,
                 populator: BaseTimeDependantPopulator = T1Population, tolerancy=1e-14):
        super().__init__(spin_system_dim, populator, tolerancy)
        self.time = time

    def compute_intensity(self, Gx, Gy, Gz, batch):
        (vector_down, vector_up), (_, _),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies, _ = batch

        intensity = mask_trans * (
                self._compute_magnitization(Gx, Gy, vector_down, vector_up, indexes) +
                self._freq_to_field(vector_down, vector_up, Gz, indexes)
        )
        return intensity

    def calculate_population_evolution(self, res_fields, mask_triu, resonance_energies, vector_down, vector_up,
                                       *args, **kwargs):
        lvl_down, lvl_up = torch.triu_indices(self.spin_system_dim,
                                              self.spin_system_dim, offset=1)
        lvl_down = lvl_down[mask_triu]
        lvl_up = lvl_up[mask_triu]
        return self.populator(res_fields, vector_down, vector_up, resonance_energies, lvl_down, lvl_up,
                              self.time, *args, **kwargs)


@dataclass
class ParamSpec:
    category: str
    dtype: torch.dtype

    def __post_init__(self):
        assert self.category in (
            "scalar", "vector", "matrix"), f"Category must be one of 'scalar', 'vector', 'matrix', got {self.category}"


class BaseSpectraCreator:
    def __init__(self, spin_system_dim, batch_dims, mesh: mesher.BaseMesh,
                 intensity_calculator: StationaryIntensitiesCalculator | None = None,
                 spectra_integrator: BaseSpectraIntegrator = SpectraIntegratorEasySpinLike(harmonic=1)):
        self.threshold = torch.tensor(1e-3)
        self.spin_system_dim = spin_system_dim
        self.mesh_size = mesh.initial_size
        self.batch_dims = batch_dims
        self.broader = Broadening()
        self.mesh = mesh
        self.spectra_integrator = spectra_integrator
        self.res_field = res_field_algorithm.ResField(output_full_eigenvector=self._get_output_eigenvector())
        self.intensity_calculator = self._get_intenisty_calculator(intensity_calculator)
        self._param_specs = self._get_param_specs()


    def _get_intenisty_calculator(self, intensity_calculator):
        if intensity_calculator is None:
            return StationaryIntensitiesCalculator(self.spin_system_dim)
        else:
            return intensity_calculator

    def _get_output_eigenvector(self) -> bool:
        return False

    def _get_param_specs(self) -> list[ParamSpec]:
        return []

    def _process_tensor(self, data_tensor: torch.Tensor):
        _, simplices = self.mesh.post_mesh
        processed = self.mesh.post_process(data_tensor.transpose(-1, -2))
        return self.mesh.to_delaunay(processed, simplices)

    def _compute_areas(self, expanded_size: torch.Tensor):
        grid, simplices = self.mesh.post_mesh
        areas = self.mesh.spherical_triangle_areas(grid, simplices)
        areas = areas.reshape(1, -1).expand(expanded_size, -1).flatten()
        return areas

    def _compute_batched_tensors(self, *args):
        batched_matrix = torch.stack(args, dim=-3)
        batched_matrix = self._process_tensor(batched_matrix)
        return batched_matrix

    def _transform_data_to_delaunay_format(
            self, res_fields: torch.Tensor, intensities: torch.Tensor, width: torch.Tensor) ->\
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

    def __call__(self,
                 sample: spin_system.MultiOrientedSample,
                 resonance_frequency: torch.Tensor,
                 fields: torch.Tensor):

        B_low  = fields[..., 0].unsqueeze(-1).expand(*self.mesh_size)
        B_high = fields[..., -1].unsqueeze(-1).expand(*self.mesh_size)

        F, Gx, Gy, Gz = sample.get_hamiltonian_terms()
        batches = self.res_field(sample, resonance_frequency, B_low, B_high, F, Gz)
        compute_out = self.compute_parameters(sample, Gx, Gy, Gz, batches)
        compute_out, fields = self._postcompute_batch_data(compute_out, fields)
        return self._finalize(compute_out, fields)

    def _postcompute_batch_data(self, compute_out: tuple, fields: torch.Tensor):
        return compute_out, fields

    def _finalize(self,
                  compute_out: tuple,
                  fields: torch.Tensor):

        _, res_fields, intensities, width, *additional_args = compute_out
        res_fields, width, intensities, areas = (
            self._transform_data_to_delaunay_format(
                res_fields, intensities, width
            )
        )

        return self.spectra_integrator.integrate(
            res_fields, width, intensities, areas, fields
        )

    def _precompute_batch_data(self, sample: spin_system.MultiOrientedSample, Gx, Gy, Gz, batch):
        intensity = self.intensity_calculator(Gx, Gy, Gz, batch)
        (vector_down, vector_up), (_, _),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies, _ = batch
        width_square = self.broader(sample, vector_down, vector_up, B_trans, indexes)
        return mask_trans, mask_triu, B_trans, intensity, width_square, indexes

    def _first_pass(self, sample, Gx, Gy, Gz, batches):
        num_pairs = (self.spin_system_dim ** 2 - self.spin_system_dim) // 2
        max_abs_intensity = torch.zeros(
            num_pairs, dtype=torch.float32, device=Gx.device
        )
        cached_batches = []
        for batch in batches:
            mask_trans, mask_triu, B_trans_batch, intensity_batch, width_square_batch, mask_indexes, *extras = \
                self._precompute_batch_data(sample, Gx, Gy, Gz, batch)

            row_idx = torch.nonzero(mask_indexes, as_tuple=False).squeeze(-1)
            col_idx = torch.nonzero(mask_triu, as_tuple=False).squeeze(-1)

            if row_idx.numel() > 0 and col_idx.numel() > 0:
                pair_max = torch.amax(intensity_batch, dim=tuple(range(intensity_batch.ndim - 1)))
                max_abs_intensity[col_idx] = torch.maximum(
                    max_abs_intensity[col_idx], pair_max
                )
            cached_batches.append(
                (mask_trans, mask_triu, B_trans_batch, intensity_batch, width_square_batch, mask_indexes, *extras)
            )

        kept_pairs = torch.nonzero(max_abs_intensity >= self.threshold, as_tuple=False).squeeze(-1)
        return cached_batches, kept_pairs

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
        batches, kept_pairs = self._first_pass(sample, Gx, Gy, Gz, batches)
        m = kept_pairs.numel()

        res_fields = torch.zeros((*config_dims, m), dtype=torch.float32, device=Gx.device)
        intensities = torch.zeros((*config_dims, m), dtype=torch.float32, device=Gx.device)
        width_square = torch.zeros((*config_dims, m), dtype=torch.float32, device=Gx.device)
        extra_tensors = self._init_extras(config_dims, m, Gx.device)

        num_pairs = (self.spin_system_dim ** 2 - self.spin_system_dim) // 2
        mask_triu_general = torch.ones(num_pairs, dtype=torch.bool)
        for batch in batches:
            mask_trans, mask_triu, B_trans_batch, intensity_batch,\
                width_square_batch, mask_indexes, *extras_batch = batch
            row_idx = torch.nonzero(mask_indexes).squeeze(-1)  # Shape [num_selected_rows]
            col_idx = torch.nonzero(mask_triu).squeeze(-1)  # Shape [num_selected_cols]
            if row_idx.numel() > 0 and col_idx.numel() > 0:
                print(col_idx)
                mask_triu_general = mask_triu_general * mask_triu
                col_base_idx, col_batch_idx = self._match_col_indexes(kept_pairs, col_idx)
                res_fields[row_idx[:, None], col_base_idx] = B_trans_batch[..., col_batch_idx]

                intensities[row_idx[:, None], col_base_idx] += intensity_batch[..., col_batch_idx]
                width_square[row_idx[:, None], col_base_idx] += width_square_batch[..., col_batch_idx]
                self._update_extras(extra_tensors, extras_batch, row_idx, col_base_idx, col_batch_idx)
        intensities = intensities / intensities.abs().max()
        width = self.broader.add_hamiltonian_straine(sample, width_square)
        return mask_triu_general, res_fields, intensities, width, *extra_tensors


# The logic with output_full_eigenvector=True must be rebuild

class TruncatedSpectraCreatorTimeResolved(BaseSpectraCreator):
    def __init__(self, spin_system_dim, batch_dims, mesh: mesher.BaseMesh, time: torch.Tensor,
                 intensity_calculator: TimeResolvedIntensitiesCalculator | None = None,
                 spectra_integrator: BaseSpectraIntegrator = SpectraIntegratorEasySpinLikeTimeResolved(harmonic=0)):
        super().__init__(spin_system_dim, batch_dims, mesh, intensity_calculator, spectra_integrator)
        self._time = time

    def _get_intenisty_calculator(self, intensity_calculator):
        if intensity_calculator is None:
            return TimeResolvedIntensitiesCalculator(self.spin_system_dim, self._time)
        else:
            return intensity_calculator

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time: torch.Tensor):
        self._time = time
        self.intensity_calculator.time = time

    def _get_param_specs(self) -> list[ParamSpec]:
        params = [
            ParamSpec("vector", torch.float32),
            ParamSpec("vector", torch.complex64),
            ParamSpec("vector", torch.complex64)
            ]
        return params

    def _get_intensity_calculator(self):
        return TimeResolvedIntensitiesCalculator(self.spin_system_dim)

    # REWRITE IT TO MAKE LOGIC BETTER
    def _precompute_batch_data(self, sample: spin_system.MultiOrientedSample, Gx, Gy, Gz, batch):
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
        return mask_trans, mask_triu, B_trans,\
            intensity, width_square, indexes, resonance_energies, vector_down, vector_up

    def _postcompute_batch_data(self, compute_out: tuple, fields: torch.Tensor):
        mask_triu, res_fields, intensities, width, *extras = compute_out
        population = self.intensity_calculator.calculate_population_evolution(
            res_fields, mask_triu, *extras
        )
        intensities = (intensities.unsqueeze(0) * population)
        return (mask_triu, res_fields, intensities, width), fields

    def _transform_data_to_delaunay_format(self, res_fields, intensities, width):
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


class CoupledSpectraCreatorTimeResolved(TruncatedSpectraCreatorTimeResolved):
    def __init__(self, spin_system_dim, batch_dims, mesh: mesher.BaseMesh, time: torch.Tensor,
                 intensity_calculator: TimeResolvedIntensitiesCalculator | None = None,
                 spectra_integrator: BaseSpectraIntegrator = SpectraIntegratorEasySpinLikeTimeResolved(harmonic=0)):
        super().__init__(spin_system_dim, batch_dims, mesh, time, intensity_calculator, spectra_integrator)

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

    def _precompute_batch_data(self, sample: spin_system.MultiOrientedSample, Gx, Gy, Gz, batch):
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
        return mask_trans, mask_triu, B_trans, intensity, width_square, indexes, resonance_energies,\
            vector_down, vector_up, vector_full





