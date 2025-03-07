import warnings
import math

import torch
from torch import nn

import constants
from mesher import Mesh
import res_field_algorithm


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

    def __call__(self, system, vector_down, vector_up, B_trans, indexes):
        strained_field = sum(
            self._compute_field_straine_square(strained_data, vector_down, vector_up, B_trans, indexes)
            for strained_data in system.build_field_dep_staine()
        )
        strained_free = sum(
            self._compute_field_free_straine_square(strained_data, vector_down, vector_up, indexes)
            for strained_data in system.build_zero_field_staine()
        )
        return strained_field + strained_free

    def add_hamiltonian_stained(self, system, squared_width):
        hamiltonian_width = system.build_hamiltonian_stained().unsqueeze(-1).square()
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
    def __init__(self, spin_system_dim, batch_dims, mesh: Mesh, interpolate: bool = True):
        self.threshold = torch.tensor(1e-3)
        self.spin_system_dim = spin_system_dim  # Как-то поменять
        self.mesh_size = mesh.size
        self.batch_dims = batch_dims
        self.intensity_calculator = IntensitiesCalculator()
        self.broader = Broadening()
        self.mesh = mesh
        self.spectra_integrator = SpectraIntegrator()
        self.res_field = res_field_algorithm.ResField()
        self.interpolate = interpolate

    def _transform_data_to_delaunay_format(self, res_fields, intensities, width):
        batched_matrix = torch.stack((res_fields, intensities, width), dim=-3)
        if self.interpolate:
            batched_matrix = self.mesh.interpolate(batched_matrix.transpose(-1, -2))
            grid, simplices = self.mesh.interpolated_mesh
        else:
            batched_matrix = batched_matrix.transpose(-1, -2)
            grid, simplices = self.mesh.initial_mesh

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

    def __call__(self, system, resonance_frequency: torch.tensor, fields: torch.Tensor):
        """
        :param system:
        :param resonance_frequency:
        :param fields: the shape is [batch_shape, n_points]
        :return:
        """
        B_low = fields[..., 0].unsqueeze(-1).expand(*self.mesh_size)
        B_high = fields[..., -1].unsqueeze(-1).expand(*self.mesh_size)

        F, Gx, Gy, Gz = system.get_hamiltonian_terms()
        batches = self.res_field(system, resonance_frequency, B_low, B_high, F, Gz)
        res_fields, intensities, width = self.compute_parameters(system, Gx, Gy, Gz, batches)
        res_fields, width, intensities, areas = self._transform_data_to_delaunay_format(
            res_fields, intensities, width)
        return self.spectra_integrator.integrate(res_fields, width, intensities, areas, fields)
        # return answer


    def _iterate_batch(self, system, Gx, Gy, Gz, batch):
        intensity = self.intensity_calculator(Gx, Gy, Gz, batch)
        (vector_down, vector_up), (_, _),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies = batch
        width_square = self.broader(system, vector_down, vector_up, B_trans, indexes)
        return mask_triu, B_trans, intensity, width_square, indexes

    #ТУТ СОЗДАЁТСЯ СИЛЬНО БОЛЬШАЯ МАТРИЦА. НУЖНО ПОМЕНЯТЬ ЛОГИКУ СОЗДАНИЯ ФИНАЛЬНЫХ МАТРИЦ!!!!!!
    #ПРИ ЭТОМ ФИНАЛЬНЫЙ РАЗМЕР МАЛЕНЬКИЙ!!
    def compute_parameters(self, system, Gx, Gy, Gz, batches):
        config_dims = (*self.batch_dims, *self.mesh_size)
        num_pairs = (self.spin_system_dim ** 2 - self.spin_system_dim) // 2
        res_fields = torch.zeros((*config_dims, num_pairs), dtype=torch.float32)
        intensities = torch.zeros((*config_dims, num_pairs), dtype=torch.float32)
        width_square = torch.zeros((*config_dims, num_pairs), dtype=torch.float32)

        for batch in batches:
            mask_triu, B_trans_batch, intensity_batch, width_square_batch, mask_indexes \
                = self._iterate_batch(system, Gx, Gy, Gz, batch)
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

        width = self.broader.add_hamiltonian_stained(system, width_square)
        return res_fields, intensities, width



# ОТСТОЙ!!! НИ ФИГА НЕ РАБОТАЕТ, НУЖНО ПЕРЕДЕЛАТЬ
class SpectraIntegrator:
    def __init__(self, harmonic: int = 0):
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        self.pi_sqrt = torch.tensor(math.sqrt(math.pi))
        self.two_sqrt = torch.tensor(math.sqrt(2.0))
        self.eps_val = torch.tensor(1e-10)
        self.natural_width = torch.tensor(1e-4)
        self.threshold = torch.tensor(1e-12)
        self.clamp = torch.tensor(1)
        self.sum_method = self._sum_method_fabric(harmonic)


    def _sum_method_fabric(self, harmonic: int = 0):
        if harmonic == 0:
            return self._absorption
        elif harmonic == 1:
            return self._derivative
        else:
            raise ValueError("Harmonic must be 0 or 1")

    def _absorption(self, arg: torch.Tensor, prefactor: torch.Tensor, width: torch.Tensor):
        log_term = -arg.square()
        log_prefactor = torch.log(prefactor)
        log_sum = torch.logsumexp(log_term + log_prefactor, dim=-1)
        return torch.exp(log_sum)

    def _derivative(self, arg: torch.Tensor, prefactor: torch.Tensor, width: torch.Tensor):
        term = torch.exp(-arg.square())
        prefactor = self.two_sqrt * arg * prefactor / width
        return torch.sum(term * prefactor, dim=-1)


    def integrate(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean:torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        r"""
        Computes the integral
            I(B) = sqrt(2/pi) * (1/width) * A_mean * I_triangle(B) * area,
        where
        :param res_fields: The resonance fields with the shape [..., M, 3]
        :param width: The width of transitions. The shape is [..., M]
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param spectral_field: The magnetic fields where spectra should be created. The shape is [...., N]
        :return: result: Tensor of shape (..., N) with the value of the integral for each B
        """
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)
        B_mean = (B1 + B2 + B3) / 3

        additional_square_width = ((B1 - B3).square() + (B2 - B3).square() + (B1 - B2).square()) / 9
        width = (self.natural_width.square() + width.square() + additional_square_width).sqrt()
        c = self.two_sqrt / width
        print(additional_square_width.sqrt().mean())
        prefactor = c * (A_mean * area) / self.pi_sqrt

        def integrand(B_val):
            arg = (B_mean - B_val) * c
            return self.sum_method(arg, prefactor, width)

        result = torch.vmap(integrand)(spectral_field)
        return result





