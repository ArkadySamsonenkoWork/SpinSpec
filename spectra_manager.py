from spectral_integration import BaseSpectraIntegrator, SpectraIntegratorExtended, SpectraIntegratorEasySpinLike
from populator_time_resolved import PopulatorTimeResolved

import torch
from torch import nn

import constants
import mesher
import res_field_algorithm
import spin_system


class StationaryPopulator:
    def __init__(self, temperature: float = 273.0):
        """
        :param temperature: temperature in K
        """
        self.temperature = temperature

    def __call__(self, energies, lvl_down, lvl_up):
        """
        :param energies: energies in Hz
        :return: population_differences
        """
        populations = nn.functional.softmax(-constants.unit_converter(energies) / self.temperature, dim=-1)
        indexes = torch.arange(populations.shape[-2], device=populations.device)
        return populations[..., indexes, lvl_up] - populations[..., indexes, lvl_down]


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


class IntensitiesCalculator:
    def __init__(self, populator=StationaryPopulator()):
        self.tolerancy = torch.tensor(1e-14)
        self.populator = populator

    def compute_matrix_element(self, vector_down, vector_up, G):
        return torch.einsum('...bi,...ij,...bj->...b', torch.conj(vector_down), G, vector_up)

    def _compute_magnitization(self, Gx, Gy, vector_down, vector_up, indexes):
        magnitization = self.compute_matrix_element(vector_down, vector_up, Gx[indexes]).square().abs() + \
                        self.compute_matrix_element(vector_down, vector_up, Gy[indexes]).square().abs()
        return magnitization

    # Как-то нужно переписать. Если 0-0, то получается большое число в интенсивности
    def _freq_to_field(self, vector_down, vector_up, Gz, indexes):
        factor_1 = self.compute_matrix_element(vector_up, vector_up, Gz[indexes])
        factor_2 = self.compute_matrix_element(vector_down, vector_down, Gz[indexes])
        return 1 / ((factor_1 - factor_2).abs() + self.tolerancy)

    def __call__(self, Gx, Gy, Gz, batch):
        (vector_down, vector_up), (lvl_down, lvl_up),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies = batch

        intensity = mask_trans * self.populator(resonance_energies, lvl_down, lvl_up) * (
                self._compute_magnitization(Gx, Gy, vector_down, vector_up, indexes) +
                self._freq_to_field(vector_down, vector_up, Gz, indexes)
        )
        return intensity

# В каждом батче тоже лучше "обрезать" малые интенсивности.
class SpectraCreator:
    def __init__(self, spin_system_dim, batch_dims, mesh: mesher.BaseMesh,
                 intensity_calculator: IntensitiesCalculator=IntensitiesCalculator(),
                 spectra_integrator: BaseSpectraIntegrator = SpectraIntegratorEasySpinLike(harmonic=1)):
        self.threshold = torch.tensor(1e-3)
        self.spin_system_dim = spin_system_dim  # Как-то поменять
        self.mesh_size = mesh.initial_size
        self.batch_dims = batch_dims
        self.intensity_calculator = intensity_calculator
        self.broader = Broadening()
        self.mesh = mesh
        self.spectra_integrator = spectra_integrator
        self.res_field = res_field_algorithm.ResField()

    def _transform_data_to_delaunay_format(self, res_fields, intensities, width):
        batched_matrix = torch.stack((res_fields, intensities, width), dim=-3)

        batched_matrix = self.mesh.post_process(batched_matrix.transpose(-1, -2))
        grid, simplices = self.mesh.post_mesh

        batched_matrix = self.mesh.to_delaunay(batched_matrix, simplices)
        expanded_size = batched_matrix.shape[-3]
        batched_matrix = batched_matrix.flatten(-3, -2)
        res_fields, intensities, width = torch.unbind(batched_matrix, dim=-3)
        width = width.mean(dim=-1)
        intensities = intensities.mean(dim=-1)

        areas = self.mesh.spherical_triangle_areas(grid, simplices)
        areas = areas.reshape(1, -1)
        areas = areas.expand(expanded_size, -1)
        areas = areas.flatten()
        return res_fields, width, intensities, areas

    def __call__(self, sample: spin_system.MultiOrientedSample, resonance_frequency: torch.tensor, fields: torch.Tensor):
        """
        :param sample:
        :param resonance_frequency:
        :param fields: the shape is [batch_shape, n_points]
        :return:
        """
        B_low = fields[..., 0].unsqueeze(-1).expand(*self.mesh_size)
        B_high = fields[..., -1].unsqueeze(-1).expand(*self.mesh_size)

        F, Gx, Gy, Gz = sample.get_hamiltonian_terms()
        batches = self.res_field(sample, resonance_frequency, B_low, B_high, F, Gz)
        res_fields, intensities, width = self.compute_parameters(sample, Gx, Gy, Gz, batches)
        res_fields, width, intensities, areas = self._transform_data_to_delaunay_format(
            res_fields, intensities, width)
        return self.spectra_integrator.integrate(res_fields, width, intensities, areas, fields)
        # return answer


    def _iterate_batch(self, sample: spin_system.MultiOrientedSample, Gx, Gy, Gz, batch):
        intensity = self.intensity_calculator(Gx, Gy, Gz, batch)
        (vector_down, vector_up), (_, _),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies = batch
        width_square = self.broader(sample, vector_down, vector_up, B_trans, indexes)
        return mask_triu, B_trans, intensity, width_square, indexes

    #ТУТ СОЗДАЁТСЯ СИЛЬНО БОЛЬШАЯ МАТРИЦА. НУЖНО ПОМЕНЯТЬ ЛОГИКУ СОЗДАНИЯ ФИНАЛЬНЫХ МАТРИЦ!!!!!!
    #ПРИ ЭТОМ ФИНАЛЬНЫЙ РАЗМЕР МАЛЕНЬКИЙ!!
    def compute_parameters(self, sample, Gx, Gy, Gz, batches):
        config_dims = (*self.batch_dims, *self.mesh_size)
        num_pairs = (self.spin_system_dim ** 2 - self.spin_system_dim) // 2
        res_fields = torch.zeros((*config_dims, num_pairs), dtype=torch.float32)
        intensities = torch.zeros((*config_dims, num_pairs), dtype=torch.float32)
        width_square = torch.zeros((*config_dims, num_pairs), dtype=torch.float32)

        for batch in batches:
            mask_triu, B_trans_batch, intensity_batch, width_square_batch, mask_indexes \
                = self._iterate_batch(sample, Gx, Gy, Gz, batch)
            row_idx = torch.nonzero(mask_indexes).squeeze(-1)  # Shape [num_selected_rows]
            col_idx = torch.nonzero(mask_triu).squeeze(-1)  # Shape [num_selected_cols]
            # Use advanced indexing to update the relevant elements
            if row_idx.numel() > 0 and col_idx.numel() > 0:
                res_fields[row_idx[:, None], col_idx] += B_trans_batch
                intensities[row_idx[:, None], col_idx] += intensity_batch
                width_square[row_idx[:, None], col_idx] += width_square_batch
        intensities = intensities.abs()
        intensities = intensities / intensities.max()
        treeshold_mask = (intensities >= self.threshold).flatten(0, -2).any(dim=0)
        intensities = intensities[..., treeshold_mask]
        res_fields = res_fields[..., treeshold_mask]
        width_square = width_square[..., treeshold_mask]
        width = self.broader.add_hamiltonian_straine(sample, width_square)
        return res_fields, intensities, width


class IntensitiesCalculatorTimeResolved:
    def __init__(self, populator=PopulatorTimeResolved()):
        self.tolerancy = torch.tensor(1e-14)
        self.populator = populator

    def compute_matrix_element(self, vector_down, vector_up, G):
        return torch.einsum('...bi,...ij,...bj->...b', torch.conj(vector_down), G, vector_up)

    def _compute_magnitization(self, Gx, Gy, vector_down, vector_up, indexes):
        magnitization = self.compute_matrix_element(vector_down, vector_up, Gx[indexes]).square().abs() + \
                        self.compute_matrix_element(vector_down, vector_up, Gy[indexes]).square().abs()
        return magnitization

    # Как-то нужно переписать. Если 0-0, то получается большое число в интенсивности
    def _freq_to_field(self, vector_down, vector_up, Gz, indexes):
        factor_1 = self.compute_matrix_element(vector_up, vector_up, Gz[indexes])
        factor_2 = self.compute_matrix_element(vector_down, vector_down, Gz[indexes])
        return 1 / ((factor_1 - factor_2).abs() + self.tolerancy)

    def __call__(self, Gx, Gy, Gz, batch):
        (vector_down, vector_up), (lvl_down, lvl_up),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies = batch

        intensity = mask_trans * (
                self._compute_magnitization(Gx, Gy, vector_down, vector_up, indexes) +
                self._freq_to_field(vector_down, vector_up, Gz, indexes)
        )
        return intensity

    def calculate_population_evolution(self, res_fields, vector_down, vector_up, resonance_energies, mask_triu):
        lvl_down, lvl_up = torch.triu_indices(4, 4, offset=1)   # REBUILD FUTHER 4 is spin system dimension
        lvl_down = lvl_down[mask_triu]
        lvl_up = lvl_up[mask_triu]
        self.populator(res_fields, vector_down, vector_up, resonance_energies, mask_triu, lvl_down, lvl_up)


class SpectraCreatorTimeResolved(SpectraCreator):
    def __init__(self, spin_system_dim, batch_dims, mesh: mesher.BaseMesh,
                 intensity_calculator: IntensitiesCalculator=IntensitiesCalculatorTimeResolved(),
                 spectra_integrator: BaseSpectraIntegrator = SpectraIntegratorEasySpinLike(harmonic=1)):
        super().__init__(spin_system_dim, batch_dims, mesh, intensity_calculator, spectra_integrator)


    def _iterate_batch(self, sample: spin_system.MultiOrientedSample, Gx, Gy, Gz, batch):
        intensity = self.intensity_calculator(Gx, Gy, Gz, batch)
        (vector_down, vector_up), (lvl_down, lvl_up),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies = batch
        width_square = self.broader(sample, vector_down, vector_up, B_trans, indexes)
        return mask_triu, B_trans, intensity, width_square, indexes, (vector_down, vector_up), resonance_energies

    #ТУТ СОЗДАЁТСЯ СИЛЬНО БОЛЬШАЯ МАТРИЦА. НУЖНО ПОМЕНЯТЬ ЛОГИКУ СОЗДАНИЯ ФИНАЛЬНЫХ МАТРИЦ!!!!!!
    #ПРИ ЭТОМ ФИНАЛЬНЫЙ РАЗМЕР МАЛЕНЬКИЙ!!
    def __call__(self, sample: spin_system.MultiOrientedSample, resonance_frequency: torch.tensor, fields: torch.Tensor):
        """
        :param sample:
        :param resonance_frequency:
        :param fields: the shape is [batch_shape, n_points]
        :return:
        """
        B_low = fields[..., 0].unsqueeze(-1).expand(*self.mesh_size)
        B_high = fields[..., -1].unsqueeze(-1).expand(*self.mesh_size)

        F, Gx, Gy, Gz = sample.get_hamiltonian_terms()
        batches = self.res_field(sample, resonance_frequency, B_low, B_high, F, Gz)
        res_fields, intensities, width, vector_down, \
            vector_up, resonance_energies, mask_triu =\
            self.compute_parameters(sample, Gx, Gy, Gz, batches)

        population = self.intensity_calculator.calculate_population_evolution(
            res_fields, vector_down, vector_up, resonance_energies, mask_triu)


        intensities = population * intensities  # REBUILD IN THE END
        res_fields, width, intensities, areas = self._transform_data_to_delaunay_format(
        res_fields, intensities, width)
        return self.spectra_integrator.integrate(res_fields, width, intensities, areas, fields)


    def compute_parameters(self, sample, Gx, Gy, Gz, batches):
        config_dims = (*self.batch_dims, *self.mesh_size)
        num_pairs = (self.spin_system_dim ** 2 - self.spin_system_dim) // 2

        res_fields = torch.zeros((*config_dims, num_pairs), dtype=torch.float32)
        intensities = torch.zeros((*config_dims, num_pairs), dtype=torch.float32)
        width_square = torch.zeros((*config_dims, num_pairs), dtype=torch.float32)

        vector_down_full = torch.zeros((*config_dims, num_pairs, self.spin_system_dim), dtype=torch.complex64)
        vector_up_full = torch.zeros((*config_dims, num_pairs, self.spin_system_dim), dtype=torch.complex64)
        resonance_energies_full = torch.zeros((*config_dims, num_pairs, self.spin_system_dim), dtype=torch.float32)
        mask_triu_general = torch.ones(num_pairs, dtype=torch.bool)

        for batch in batches:
            mask_triu, B_trans_batch, intensity_batch, width_square_batch, mask_indexes,\
                (vector_down, vector_up), resonance_energies = self._iterate_batch(sample, Gx, Gy, Gz, batch)
            row_idx = torch.nonzero(mask_indexes).squeeze(-1)  # Shape [num_selected_rows]
            col_idx = torch.nonzero(mask_triu).squeeze(-1)  # Shape [num_selected_cols]
            # Use advanced indexing to update the relevant elements
            if row_idx.numel() > 0 and col_idx.numel() > 0:
                mask_triu_general = mask_triu_general * mask_triu
                res_fields[row_idx[:, None], col_idx] += B_trans_batch
                intensities[row_idx[:, None], col_idx] += intensity_batch
                width_square[row_idx[:, None], col_idx] += width_square_batch
                vector_down_full[row_idx[:, None], col_idx, :] += vector_down
                vector_up_full[row_idx[:, None], col_idx, :] += vector_up
                resonance_energies_full[row_idx[:, None], col_idx, :] += resonance_energies

        intensities = intensities.abs()
        intensities = intensities / intensities.max()
        treeshold_mask = (intensities >= self.threshold).flatten(0, -2).any(dim=0)
        intensities = intensities[..., treeshold_mask]
        res_fields = res_fields[..., treeshold_mask]
        width_square = width_square[..., treeshold_mask]

        vector_down_full = vector_down_full[..., treeshold_mask, :]
        vector_up_full = vector_up_full[..., treeshold_mask, :]
        resonance_energies_full = resonance_energies_full[..., treeshold_mask, :]
        width = self.broader.add_hamiltonian_straine(sample, width_square)


        return res_fields, intensities, width, vector_down_full,\
            vector_up_full, resonance_energies_full, mask_triu_general







