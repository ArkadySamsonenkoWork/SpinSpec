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
    def __init__(self, spin_system_dim, batch_dims, mesh: Mesh, interpolate: bool = False):
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



# КАК_ТО Я ЗАСТАВИЛ ЕГО РАБОТАТЬ!!!!!!!
class SpectraIntegratorUnbehabiour:
    def __init__(self, harmonic: int = 0):
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        self.pi_sqrt = torch.tensor(math.sqrt(math.pi))
        self.two_sqrt = torch.tensor(math.sqrt(2.0))

        self.natural_width = 2e-5
        self.eps_val = torch.tensor(1e-10)
        self.threshold = torch.tensor(1e-12)
        self.clamp = torch.tensor(10)
        self.sum_method = self._sum_method_fabric(harmonic)
        self.equality_threshold = 1e-2

    def _sum_method_fabric(self, harmonic: int = 0):
        if harmonic == 0:
            return self._absorption
        elif harmonic == 1:
            return self._derivative
        else:
            raise ValueError("Harmonic must be 0 or 1")

    def _absorption(self, x: torch.Tensor, c_val: torch.Tensor, width_val: torch.Tensor):
        arg = torch.clamp(c_val * x, -self.clamp, self.clamp)  # Prevent large values
        #arg = c_val * x
        erf_val = torch.erf(arg)
        exp_val = torch.exp(torch.clamp(-arg.square(), min=-50))  # Prevent underflow
        return (c_val * x * erf_val + (1 / self.pi_sqrt) * exp_val)

    def _derivative(self, x: torch.Tensor, c_val: torch.Tensor, width_val: torch.Tensor):
        arg = torch.clamp(c_val * x, -self.clamp, self.clamp)
        return torch.erf(arg)

    def integrate(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean: torch.Tensor,
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
        width = width
        width = self.natural_width + width
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)
        B1 = B1 + width / 50
        B3 = B3 - width / 50

        d13 = (B1 - B3) / width
        d23 = (B2 - B3) / width
        d12 = (B1 - B2) / width


        c = self.two_sqrt / width

        denominator_full = d12 * d23 * d13

        selfafe_denom_full = torch.where(torch.abs(denominator_full) < self.threshold,
                                      denominator_full + self.eps_val,
                                      denominator_full)

        def integrand_full(B_val):
            arg_1 = (B1 - B_val) / width
            arg_2 = (B2 - B_val) / width
            arg_3 = (B3 - B_val) / width

            X1 = self.sum_method(B1 - B_val, c, width)
            X2 = self.sum_method(B2 - B_val, c, width)
            X3 = self.sum_method(B3 - B_val, c, width)

            # num = X1 / (d13 * d12) - X2 / (d12 * d23) + X3 / (d13 * d23)
            num = (X1 * d23 - X2 * d13 + X3 * d12)
            num = torch.where((arg_1.square() + arg_2.square() + arg_3.square()).sqrt() < 1000, num, 0.0)  # IT HEALS BEHAVIOUR AT INFTY

            log_num = torch.log(torch.abs(num) + self.eps_val)
            log_den = torch.log(torch.abs(selfafe_denom_full) + + self.eps_val)
            ratio = num / selfafe_denom_full
            print(arg_1)
            print(ratio)

            # Recover the ratio while keeping track of the sign
            #ratio = torch.sign(num) * torch.exp(log_num - log_den)

            return (ratio * self.two_sqrt * (A_mean * area) / width).sum(dim=-1)

        result = torch.vmap(integrand_full)(spectral_field)
        return result


class SpectraIntegrator:
    def __init__(self, harmonic: int = 0):
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        self.pi_sqrt = torch.tensor(math.sqrt(math.pi))
        self.two_sqrt = torch.tensor(math.sqrt(2.0))

        self.natural_width = 1e-5
        self.eps_val = torch.tensor(1e-12)
        self.threshold = torch.tensor(1e-6)
        self.clamp = torch.tensor(50)
        self.sum_method = self._sum_method_fabric(harmonic)
        self.equality_threshold = 1e-2


    def _sum_method_fabric(self, harmonic: int = 0):
        if harmonic == 0:
            return self._absorption
        elif harmonic == 1:
            return self._derivative
        else:
            raise ValueError("Harmonic must be 0 or 1")

    def _absorption(self, x: torch.Tensor, c_val: torch.Tensor, width_val: torch.Tensor):
        arg = torch.clamp(c_val * x, -self.clamp, self.clamp)  # Prevent large values
        erf_val = torch.erf(arg)
        exp_val = torch.exp(torch.clamp(-arg.square(), min=-50))  # Prevent underflow
        return (c_val * x * erf_val + (1 / self.pi_sqrt) * exp_val)


    def _derivative(self, x: torch.Tensor, c_val: torch.Tensor, width_val: torch.Tensor):
        arg = torch.clamp(c_val * x, -self.clamp, self.clamp)
        return torch.erf(arg)

    def _get_equality_masks(self, B1, B2, B3, width):
        diff12 = torch.abs(B1 - B2) / width
        diff13 = torch.abs(B1 - B3) / width
        diff23 = torch.abs(B2 - B3) / width

        eq12 = diff12 < self.equality_threshold
        eq13 = diff13 < self.equality_threshold
        eq23 = diff23 < self.equality_threshold

        mask_all_diff = ~(eq12 | eq13 | eq23)

        # Case 2: Two are equal, third is different
        mask_two_eq = (eq12 & ~eq13) | (eq13 & ~eq12) | (eq23 & ~eq12)

        # Case 3: All three are equal
        mask_all_eq = eq12 & eq13 & eq23

        return mask_all_diff, mask_two_eq, mask_all_eq

    def _two_equality_case(self, B1, B2, B3, width):
        diff12 = torch.abs(B1 - B2) / width
        diff13 = torch.abs(B1 - B3) / width
        diff23 = torch.abs(B2 - B3) / width

        eq12 = diff12 < self.equality_threshold
        eq13 = diff13 < self.equality_threshold
        eq23 = diff23 < self.equality_threshold

        B1, B3 = torch.where(eq12, B3, B1), torch.where(eq12, B1, B3)
        B1, B2 = torch.where(eq13, B2, B1), torch.where(eq13, B1, B2)

        return B1, B2, B3

    def _apply_mask(self, B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor,
                  width: torch.Tensor, A_mean:torch.Tensor,
                  area: torch.Tensor, mask):
        B1 = B1[mask]
        B2 = B2[mask]
        B3 = B3[mask]
        area = area[mask]
        A_mean = A_mean[mask]
        width = width[mask]
        return B1, B2, B3, width, area, A_mean


    def all_different_case(self, B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor,
                  width: torch.Tensor, A_mean:torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor, mask_all_diff):
        B1, B2, B3, width, area, A_mean = self._apply_mask(B1, B2, B3, width, A_mean, area, mask_all_diff)

        d13 = (B1 - B3) / width
        d23 = (B2 - B3) / width
        d12 = (B1 - B2) / width

        c = self.two_sqrt / width
        denominator = d12 * d23 * d13
        safe_denom = torch.where(torch.abs(denominator) < self.threshold,
                                       denominator + self.eps_val,
                                       denominator)
        def integrand(B_val):
            arg_1 = (B1 - B_val) / width
            arg_2 = (B2 - B_val) / width
            arg_3 = (B3 - B_val) / width
            X1 = self.sum_method(B1 - B_val, c, width)
            X2 = self.sum_method(B2 - B_val, c, width)
            X3 = self.sum_method(B3 - B_val, c, width)

            num = (X1 * d23 - X2 * d13 + X3 * d12)
            #num = torch.where((arg_1.square() + arg_2.square() + arg_3.square()).sqrt() < 20, num, 0.0)  # IT HEALS BEHAVIOUR AT INFTY

            #log_num = torch.log(torch.abs(num) + self.eps_val)
            #log_den = torch.log(torch.abs(safe_denom))
            ratio = num / safe_denom
            #ratio = torch.sign(num) * torch.exp(log_num - log_den)
            return (ratio * (1 / self.two_sqrt) * (A_mean * area) * (1 / width)).sum(dim=-1)

        result = torch.vmap(integrand)(spectral_field)
        return result


    def two_eq_case(self, B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor,
                           width: torch.Tensor, A_mean: torch.Tensor,
                           area: torch.Tensor, spectral_field: torch.Tensor, mask_two_equel):
        B1, B2, B3, width, area, A_mean = self._apply_mask(B1, B2, B3, width, A_mean, area, mask_two_equel)
        B1, B2, B3 = self._two_equality_case(B1, B2, B3, width)  # B2 == B3

        d13 = (B1 - B3) / width
        d12 = (B1 - B2) / width

        c = self.two_sqrt / width
        denominator = d12 * d13

        safe_denom = torch.where(torch.abs(denominator) < self.threshold,
                                 denominator + self.eps_val,
                                 denominator)

        def F(x1, x2, c_val, width):
            arg = torch.clamp(c_val * x2, -self.clamp, self.clamp)  # Prevent large values
            erf_val = torch.erf(arg)
            exp_val = torch.exp(torch.clamp(-arg.square(), min=-50))  # Prevent underflow
            return (c_val * x1 * erf_val + (1 / self.pi_sqrt) * exp_val)

        def integrand(B_val):
            X1 = F(B1 - B_val, B1 - B_val, c, width)
            X2 = F(B1 - B_val, B2 - B_val, c, width)
            num = X1 - X2

            eps = self.eps_val
            log_num = torch.log(torch.abs(num) + eps)
            log_den = torch.log(torch.abs(safe_denom))

            ratio = torch.sign(num) * torch.exp(log_num - log_den)
            return (ratio * (1 / (2 * self.two_sqrt)) * (A_mean * area) * (1 / width)).sum(dim=-1)
        result = torch.vmap(integrand)(spectral_field)
        return result


    def all_eq_case(self, B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor,
                    width: torch.Tensor, A_mean: torch.Tensor,
                    area: torch.Tensor, spectral_field: torch.Tensor, mask_all_equel):
        B1, B2, B3, width, area, A_mean = self._apply_mask(B1, B2, B3, width, A_mean, area, mask_all_equel)
        B0 = (B1 + B2 + B3) / 3
        d23 = (B2 - B3)
        d13 = (B1 - B3)
        d12 = (B1 - B2)
        width = (width.square() + (d12.square() + d13.square() + d23.square()) / 9).sqrt()
        c = self.two_sqrt / width
        def integrand(B_val):
            arg = (B0 - B_val) * c
            term = (self.two_sqrt / self.pi_sqrt) * torch.exp(-arg**2) * (1 / width)
            return (term * (A_mean * area)).sum(dim=-1)

        result = torch.vmap(integrand)(spectral_field)
        return result

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
        width = width
        width = self.natural_width + width
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)

        mask_all_diff, mask_two_eq, mask_all_eq = self._get_equality_masks(B1, B2, B3, width)
        #mask_two_eq = mask_two_eq * 0
        #mask_all_eq = mask_all_eq * 0
        #mask_all_diff = mask_all_diff * 0


        print(mask_two_eq.sum())
        print(mask_all_eq.sum())
        print(mask_all_diff.sum())

        result =\
            self.all_different_case(B1, B2, B3, width, A_mean, area, spectral_field, mask_all_diff) + \
            self.two_eq_case(B1, B2, B3, width, A_mean, area, spectral_field, mask_two_eq) + \
            self.all_eq_case(B1, B2, B3, width, A_mean, area, spectral_field, mask_all_eq)

        return result





