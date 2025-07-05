import warnings
import typing as tp
from dataclasses import dataclass

from spectral_integration import BaseSpectraIntegrator,\
    SpectraIntegratorEasySpinLike,\
    SpectraIntegratorEasySpinLikeTimeResolved
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
    return torch.einsum('...bi,...ij,...bj->...b', torch.conj(vector_down), G, vector_up)


class PostSpectraProcessing:
    def __init__(self, gauss: torch.Tensor, lorentz: torch.Tensor):
        """
        :param gauss: The gauss parameter. The shape is [batch_size] or []
        :param lorentz: The gauss parameter. The shape is [batch_size] or []
        """
        self.gauss = gauss
        self.lorentz = lorentz
        self._broading_method = self._broading_fabric(gauss, lorentz)

    def _skip_broader(self, magnetic_fields: torch.Tensor, spec: torch.Tensor):
        return spec

    def _broading_fabric(self, gauss, lorentz):
        if (gauss == 0) and (lorentz == 0):
            return self._skip_broader

        elif (gauss != 0) and (lorentz == 0):
            return self._gauss_broader

        elif (gauss == 0) and (lorentz != 0):
            return self._lorentz_broader

        else:
            return self._voigt_broader

    def __call__(self, magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        :param magnetic_field: Tensor of shape [batch, N]
        :param spec: Spectrum tensor of shape [batch, ..., N]
        :return: Broadended spectrum, same shape as spec
        """
        return self._broading_method(magnetic_field, spec)

    def _build_kernel(self, magnetic_field: torch.Tensor, fwhm_gauss: torch.Tensor, fwhm_lorentz: torch.Tensor):
        """Construct Voigt kernel on grid for each sample."""
        dH = magnetic_field[..., 1] - magnetic_field[..., 0]
        N = magnetic_field.shape[-1]
        idx = torch.arange(N, device=magnetic_field.device) - N//2
        x = idx * dH

        sigma = fwhm_gauss / (2 * (2 * torch.log(torch.tensor(2.0)))**0.5)

        gamma = fwhm_lorentz / 2
        # gaussian kernel
        G = torch.exp(-0.5 * (x / sigma)**2) / (sigma * (2 * torch.pi)**0.5)

        # lorentzian kernel
        L = (gamma / torch.pi) / (x**2 + gamma**2)

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

    def _gauss_broader(self, magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        batch = magnetic_field.shape[0]
        fwhm_gauss = self.gauss.expand(batch)
        fwhm_lorentz = torch.zeros_like(fwhm_gauss)
        kernel = self._build_kernel(magnetic_field, fwhm_gauss, fwhm_lorentz)
        return self._apply_convolution(spec, kernel)

    def _lorentz_broader(self, magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        batch = magnetic_field.shape[0]
        fwhm_gauss = torch.zeros_like(self.lorentz.expand(batch))
        fwhm_lorentz = self.lorentz.expand(batch)
        kernel = self._build_kernel(magnetic_field, fwhm_gauss, fwhm_lorentz)
        return self._apply_convolution(spec, kernel)

    def _voigt_broader(self, magnetic_field: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        batch = magnetic_field.shape[0]
        fwhm_gauss = self.gauss.expand(batch)
        fwhm_lorentz = self.lorentz.expand(batch)
        kernel = self._build_kernel(magnetic_field, fwhm_gauss, fwhm_lorentz)
        return self._apply_convolution(spec, kernel)


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
    def __init__(self, spin_system_dim: int, populator: tp.Callable, tolerancy=1e-12):
        self.tolerancy = torch.tensor(tolerancy)
        self.populator = populator
        self.spin_system_dim = spin_system_dim

    def _compute_magnitization(self, Gx, Gy, vector_down, vector_up, indexes):
        magnitization = compute_matrix_element(vector_down, vector_up, Gx[indexes]).square().abs() + \
                        compute_matrix_element(vector_down, vector_up, Gy[indexes]).square().abs()
        return magnitization

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
                self._compute_magnitization(Gx, Gy, vector_down, vector_up, indexes)
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
                self._compute_magnitization(Gx, Gy, vector_down, vector_up, indexes)
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
                 spectra_integrator: BaseSpectraIntegrator = SpectraIntegratorEasySpinLike(harmonic=1),
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(
                     torch.tensor(0), torch.tensor(0))
                 ):
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

        self.tolerancy = torch.tensor(1e-12)
        self._vacuum_g_factor = torch.tensor(2.002)

        self.post_spectra_processor = post_spectra_processor

    def _get_intenisty_calculator(self, intensity_calculator):
        if intensity_calculator is None:
            return StationaryIntensitiesCalculator(self.spin_system_dim)
        else:
            return intensity_calculator

    def _freq_to_field(self, vector_down, vector_up, Gz, indexes):
        """Compute frequency-to-field contribution"""
        factor_1 = compute_matrix_element(vector_up, vector_up, Gz[indexes])
        factor_2 = compute_matrix_element(vector_down, vector_down, Gz[indexes])
        diff = (factor_1 - factor_2).abs()
        safe_diff = torch.where(diff < self.tolerancy, self.tolerancy, diff)
        safe_diff = constants.unit_converter(safe_diff, "Hz_to_T_e") / self._vacuum_g_factor # Must be rebuild to change broading units
        return safe_diff.reciprocal()


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

    def _final_mask(self, res_fields: torch.Tensor, width: torch.Tensor,
                    intensities: torch.Tensor, areas: torch.Tensor):
        max_intensity = torch.amax(abs(intensities), dim=-1, keepdim=True)
        mask = ((intensities / max_intensity) > self.threshold).any(dim=tuple(range(intensities.dim() - 1)))
        #mask = (intensities > self.threshold)

        intensities = intensities[..., mask]
        width = width[..., mask]
        res_fields = res_fields[..., mask, :]
        areas = areas[..., mask]
        return res_fields, width, intensities, areas


    def _finalize(self,
                  compute_out: tuple,
                  fields: torch.Tensor):

        _, res_fields, intensities, width, *additional_args = compute_out
        res_fields, width, intensities, areas = (
            self._transform_data_to_delaunay_format(
                res_fields, intensities, width
            )
        )

        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)
        spec = self.spectra_integrator.integrate(
            res_fields, width, intensities, areas, fields
        )
        return self.post_spectra_processor(fields, spec)

    def _precompute_batch_data(self, sample: spin_system.MultiOrientedSample, Gx, Gy, Gz, batch):
        intensity = self.intensity_calculator(Gx, Gy, Gz, batch)
        (vector_down, vector_up), (_, _),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies, _ = batch

        freq_to_field_val = self._freq_to_field(vector_down, vector_up, Gz, indexes)

        width_square = self.broader(sample, vector_down, vector_up, B_trans, indexes)
        return mask_trans, mask_triu, B_trans, intensity, width_square, indexes, freq_to_field_val

    def _filter_by_max(self, occurrences_max, global_max):
        updated_occurrences = []
        for bi, row_idx, col_idx, pair_max in occurrences_max:
            keep_local = torch.nonzero(pair_max / global_max >= self.threshold, as_tuple=False).squeeze(-1)
            new_cold_idx = col_idx[keep_local]
            if new_cold_idx.numel() > 0:
                updated_occurrences.append((bi, row_idx, col_idx[keep_local], keep_local))
        return updated_occurrences

    # The logic can be broken. I do not really know for difficult examples!!
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
        row_to_tuples = {}  # row_id -> list of prior tuple-indices
        assigned_global: list[torch.Tensor] = []
        out: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        max_global = -1

        for i, (batch_idx, rows, cols, keep_pairs) in enumerate(occurrences):
            conflicts = set()
            for r in rows.tolist():
                conflicts.update(row_to_tuples.get(r, []))

            base_gh = torch.argsort(torch.argsort(cols))

            # 3) collect occupied slots only from those conflicts
            occupied = set()
            for j in conflicts:
                occupied.update(assigned_global[j].tolist())

            # 4) if there’s conflict, shift to avoid _just_ those slots
            global_idx = base_gh.clone()
            if occupied:
                s = 0
                while True:
                    shifted = (base_gh + s).tolist()
                    if not any(x in occupied for x in shifted):
                        global_idx = base_gh + s
                        break
                    s += 1

            # 5) record and update max
            out.append((batch_idx, rows, cols, global_idx, keep_pairs))
            assigned_global.append(global_idx)
            max_global = max(max_global, int(global_idx.max()))

            # 6) register rows
            for r in rows.tolist():
                row_to_tuples.setdefault(r, []).append(i)

        return out, max_global + 1

    def _first_pass(self, sample, Gx, Gy, Gz, batches):
        cached_batches = []
        occurences_max = []

        global_max = torch.tensor(0)
        for bi, batch in enumerate(batches):
            mask_trans, mask_triu, B_trans_batch, intensity_batch, width_square_batch, mask_indexes,\
                freq_to_field_val, *extras = \
                self._precompute_batch_data(sample, Gx, Gy, Gz, batch)

            row_idx = torch.nonzero(mask_indexes, as_tuple=False).squeeze(-1)
            col_idx = torch.nonzero(mask_triu, as_tuple=False).squeeze(-1)

            if row_idx.numel() > 0 and col_idx.numel() > 0:

                pair_max = torch.amax(torch.abs(intensity_batch), dim=tuple(range(intensity_batch.ndim - 1)))
                global_max = torch.max(global_max, torch.max(pair_max))

                cached_batches.append(
                    (mask_trans,
                     mask_triu, B_trans_batch,
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
        batches, occurrences = self._first_pass(sample, Gx, Gy, Gz, batches)
        occurrences, max_columns = self._assign_global_indexes(occurrences)

        res_fields = torch.zeros((*config_dims, max_columns), dtype=torch.float32, device=Gx.device)
        intensities = torch.zeros((*config_dims, max_columns), dtype=torch.float32, device=Gx.device)
        width_square = torch.zeros((*config_dims, max_columns), dtype=torch.float32, device=Gx.device)
        freq_to_field_global = torch.zeros((*config_dims, max_columns), dtype=torch.float32, device=Gx.device)

        extra_tensors = self._init_extras(config_dims, max_columns, Gx.device)

        num_pairs = (self.spin_system_dim ** 2 - self.spin_system_dim) // 2
        mask_triu_general = torch.zeros(num_pairs, dtype=torch.bool)

        for bi, _, col_batch_idx, col_base_idx, keep_local in occurrences:
            mask_trans, mask_triu, B_trans_batch, intensity_batch,\
                width_square_batch, mask_indexes, freq_to_field_val, *extras_batch = batches[bi]
            row_idx = torch.nonzero(mask_indexes).squeeze(-1)

            if row_idx.numel() > 0 and col_base_idx.numel() > 0:
                mask_triu_general[..., col_batch_idx] = True
                res_fields[row_idx[:, None], col_base_idx] = B_trans_batch[..., keep_local]

                intensities[row_idx[:, None], col_base_idx] += intensity_batch[..., keep_local]
                width_square[row_idx[:, None], col_base_idx] += width_square_batch[..., keep_local]

                freq_to_field_global[row_idx[:, None], col_base_idx] += freq_to_field_val[..., keep_local]

                self._update_extras(extra_tensors, extras_batch, row_idx, col_base_idx, keep_local)

        intensities *= freq_to_field_global
        intensities = intensities / intensities.abs().max()
        width = self.broader.add_hamiltonian_straine(sample, width_square) * freq_to_field_global
        return mask_triu_general, res_fields, intensities, width, *extra_tensors


# The logic with output_full_eigenvector=True must be rebuild
class TruncatedSpectraCreatorTimeResolved(BaseSpectraCreator):
    def __init__(self, spin_system_dim, batch_dims, mesh: mesher.BaseMesh, time: torch.Tensor,
                 intensity_calculator: TimeResolvedIntensitiesCalculator | None = None,
                 spectra_integrator: BaseSpectraIntegrator = SpectraIntegratorEasySpinLikeTimeResolved(harmonic=0),
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(
                    torch.tensor(0), torch.tensor(0))):
        super().__init__(
            spin_system_dim, batch_dims, mesh, intensity_calculator, spectra_integrator, post_spectra_processor
        )
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

        freq_to_field_val = self._freq_to_field(vector_down, vector_up, Gz, indexes)
        return mask_trans, mask_triu, B_trans,\
            intensity, width_square, indexes, freq_to_field_val, resonance_energies, vector_down, vector_up

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
                 spectra_integrator: BaseSpectraIntegrator = SpectraIntegratorEasySpinLikeTimeResolved(harmonic=0),
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(
                     torch.tensor(0), torch.tensor(0))):
        super().__init__(
            spin_system_dim, batch_dims, mesh, time, intensity_calculator, spectra_integrator, post_spectra_processor)

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

        freq_to_field_val = self._freq_to_field(vector_down, vector_up, Gz, indexes)
        return mask_trans, mask_triu, B_trans, intensity, width_square, indexes, freq_to_field_val, resonance_energies,\
            vector_down, vector_up, vector_full





