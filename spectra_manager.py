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

    def compute_element(self, vector, G):
        return torch.einsum('...bi,...ij,...bj->...b', torch.conj(vector), G, vector).real

    def _compute_field_straine_square(self, straine, vector_down, vector_up, B_trans, indexes):
        return (B_trans * (
                self.compute_element(vector_up, straine[indexes]) -
                self.compute_element(vector_down, straine[indexes])
        )).square()

    def _compute_field_free_straine_square(self, straine, vector_down, vector_up, indexes):
        return (
                self.compute_element(vector_up, straine[indexes]) -
                self.compute_element(vector_down, straine[indexes])
        ).square()

    def __call__(self, system, vector_down, vector_up, B_trans, indexes):
        straine_field = sum(
            self._compute_field_straine_square(straine, vector_down, vector_up, B_trans, indexes) for straine in system.build_field_dep_stained()
        )
        straine_free = sum(
            self._compute_field_free_straine_square(straine, vector_down, vector_up, indexes) for straine in system.build_zero_field_stained()
        )
        return straine_field + straine_free

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
    def __init__(self, spin_system_dim, batch_dims, mesh: Mesh):
        self.threshold = torch.tensor(1e-3)
        self.spin_system_dim = spin_system_dim  # Как-то поменять
        self.batch_dims = batch_dims
        self.intensity_calculator = IntensitiesCalculator()
        self.broader = Broadening()
        self.mesh = mesh
        self.spectra_integrator = SpectraIntegrator()
        self.res_field = res_field_algorithm.ResField()

    # ТОЖЕ ПЕРЕДЕЛАТЬ ЛОГИКУ
    def __call__(self, system, resonance_frequency: torch.tensor, fields: torch.Tensor):
        """
        :param system:
        :param resonance_frequency:
        :param fields: the shape is [batch_shape, n_points]
        :return:
        """
        mesh_size = self.mesh.size
        B_low = fields[..., 0].unsqueeze(-1).expand(*mesh_size)
        B_high = fields[..., -1].unsqueeze(-1).expand(*mesh_size)

        F, Gx, Gy, Gz = system.get_hamiltonian_terms()
        batches = self.res_field(system, resonance_frequency, B_low, B_high, F, Gz)
        res_fields, intensities, width = self.compute_parameters(system, Gx, Gy, Gz, batches)
        #return res_fields, intensities, width
        width = self.broader.add_hamiltonian_stained(system, width)
        #res_fields = self.mesh.interpolate(res_fields.transpose(-1, -2))
        #intensities = self.mesh.interpolate(intensities.transpose(-1, -2))
        #width = self.mesh.interpolate(width.transpose(-1, -2))

        res_fields = res_fields.transpose(-1, -2)
        intensities = intensities.transpose(-1, -2)
        width = width.transpose(-1, -2)

        res_fields = self.mesh.to_delaunay(res_fields)
        intensities = self.mesh.to_delaunay(intensities).mean(dim=-1)
        width = self.mesh.to_delaunay(width).mean(dim=-1)

        #interpolation_grid = torch.tensor([[val[0], val[1]] for val in self.mesh.interpolation_grid])  #
        interpolation_grid = torch.tensor([[val[0], val[1]] for val in self.mesh.initial_grid])
        areas = self.mesh.spherical_triangle_areas(interpolation_grid, self.mesh.initial_simpleces)

        expanded_size = res_fields.shape[-3]
        res_fields = res_fields.flatten(-3, -2)

        width = width.flatten(-2, -1)
        intensities = intensities.flatten(-2, -1)

        areas = areas.reshape(1, -1)
        areas = areas.expand(expanded_size, -1)
        areas = areas.flatten()
        return self.spectra_integrator.integrate(res_fields, width, intensities, areas, fields)
        # return answer


    def _iterate_batche(self, system, Gx, Gy, Gz, batch):
        intensity = self.intensity_calculator(Gx, Gy, Gz, batch)
        (vector_down, vector_up), (_, _),\
            B_trans, mask_trans, mask_triu, indexes, resonance_energies = batch
        width_square = self.broader(system, vector_down, vector_up, B_trans, indexes)
        return mask_triu, B_trans, intensity, width_square, indexes

    #ТУТ СОЗДАЁТСЯ СИЛЬНО БОЛЬШАЯ МАТРИЦА. НУЖНО ПОМЕНЯТЬ ЛОГИКУ СОЗДАНИЯ ФИНАЛЬНЫХ МАТРИЦ!!!!!!
    #ПРИ ЭТОМ ФИНАЛЬНЫЙ РАЗМЕР МАЛЕНЬКИЙ!!
    def compute_parameters(self, system, Gx, Gy, Gz, batches):
        num_pairs = (self.spin_system_dim ** 2 - self.spin_system_dim) // 2
        res_fields = torch.zeros((*self.batch_dims, num_pairs), dtype=torch.float32)
        intensities = torch.zeros((*self.batch_dims, num_pairs), dtype=torch.float32)
        width_square = torch.zeros((*self.batch_dims, num_pairs), dtype=torch.float32)
        for batch in batches:
            mask_triu, B_trans_batch, intensity_batch, width_square_batch, mask_indexes \
                = self._iterate_batche(system, Gx, Gy, Gz, batch)
            row_idx = torch.nonzero(mask_indexes).squeeze(-1)  # Shape [num_selected_rows]
            col_idx = torch.nonzero(mask_triu).squeeze(-1)  # Shape [num_selected_cols]
            # Use advanced indexing to update the relevant elements
            if row_idx.numel() > 0 and col_idx.numel() > 0:
                res_fields[row_idx[:, None], col_idx] += B_trans_batch
                intensities[row_idx[:, None], col_idx] += intensity_batch
                width_square[row_idx[:, None], col_idx] += width_square_batch

            #intensities[indexes][..., mask_triu] += 1123/
        intensities = intensities.abs()
        intensities = intensities / intensities.max()
        treeshold_mask = (intensities >= self.threshold).flatten(0, -2).any(dim=0)
        #print(treeshold_mask.shape)
        intensities = intensities[..., treeshold_mask]
        res_fields = res_fields[..., treeshold_mask]
        width_square = width_square[..., treeshold_mask]
        return res_fields, intensities, width_square

class SpectraIntegrator:
    def __init__(self):
        self.sqrt_pi = torch.tensor(math.sqrt(math.pi))
        self.two_sqrt = torch.tensor(math.sqrt(2.0))
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))
        self.eps_val = torch.tensor(1e-12)
        self.const_factor = torch.sqrt(torch.tensor(2.0 / math.pi))


    def __integrate(self, res_fields, width, A_mean, area, B):
        r"""
        Computes the integral
            I(B) = sqrt(2/pi) * (1/width) * A_mean * I_triangle(B) * area,
        where
        :param res_fields: The resonance fields with the shape [..., M, 3]
        :param width: The width of transitions. The shape is [..., M]
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param B: The magnetic fields where spectra should be created.
        :return: result: Tensor of shape (..., N) with the value of the integral for each B
        """
        #B = B[:10]

        width = width / 1000000
        width = self.eps_val + width * constants.PLANCK / constants.BOHR
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)

        d13 = (B1 - B3).unsqueeze(-1)
        d23 = (B2 - B3).unsqueeze(-1)
        d12 = (B1 - B2).unsqueeze(-1)

        c = self.two_sqrt / width
        c_exp = c.unsqueeze(-1)
        width_exp = width.unsqueeze(-1)
        def F(x):
            return x * torch.erf(c_exp * x) + (width_exp / (self.two_sqrt * self.sqrt_pi)) * torch.exp(
                - (c_exp * x) ** 2)


        X1 = F(B1.unsqueeze(-1) - B)
        X2 = F(B2.unsqueeze(-1) - B)
        X3 = F(B3.unsqueeze(-1) - B)
        term = (X1 * d23 - X2 * d13 + X3 * d12) / (d12 * d23 * d13 + self.eps_val)


        factor = A_mean
        result = torch.einsum('...mn,m->...n', term * factor.unsqueeze(-1), area)
        return result


    def integrate(self, res_fields, width, A_mean, area, B):
        r"""
        Computes the integral
            I(B) = sqrt(2/pi) * (1/width) * A_mean * I_triangle(B) * area,
        where
        :param res_fields: The resonance fields with the shape [..., M, 3]
        :param width: The width of transitions. The shape is [..., M]
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param B: The magnetic fields where spectra should be created.
        :return: result: Tensor of shape (..., N) with the value of the integral for each B
        """
        #B = B[:10]
        width = width / 1000000
        width = self.eps_val + width * constants.PLANCK / constants.BOHR
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)

        d13 = (B1 - B3).unsqueeze(-1)
        d23 = (B2 - B3).unsqueeze(-1)
        d12 = (B1 - B2).unsqueeze(-1)

        c = self.two_sqrt / width
        c_exp = c.unsqueeze(-1)
        width_exp = width.unsqueeze(-1)
        CLAMP_MIN = -50.0
        CLAMP_MAX = 50.0

        def F(x):
            # x will be broadcast to shape compatible with c_exp
            arg = torch.clamp(c_exp * x, CLAMP_MIN, CLAMP_MAX)
            erf_val = torch.erf(arg)
            exp_val = torch.exp(- arg**2)
            return x * erf_val + (width_exp / (self.two_sqrt * self.sqrt_pi)) * exp_val

        # Evaluate function F at the differences between resonance fields and magnetic fields
        X1 = F(B1.unsqueeze(-1) - B)
        X2 = F(B2.unsqueeze(-1) - B)
        X3 = F(B3.unsqueeze(-1) - B)

        # Compute numerator and denominator for the term of the integral
        numerator = (X1 * d23 - X2 * d13 + X3 * d12)
        denominator = d12 * d23 * d13

        # Avoid division by a near-zero denominator using a conditional threshold
        threshold = 1e-12
        safe_denominator = torch.where(torch.abs(denominator) < self.eps_val,
                                       denominator + self.eps_val,
                                       denominator)
        term = numerator / safe_denominator

        # Multiply by the intensity factor and perform the weighted sum over transitions
        result = torch.einsum('...mn,m->...n', term * A_mean.unsqueeze(-1), area)
        return result


    def dfgntegrate(self, res_fields, width, A_mean, area, B):
        r"""
        Computes the integral
            I(B) = sqrt(2/pi) * (1/width) * A_mean * I_triangle(B) * area,
        where
        :param res_fields: The resonance fields with the shape [..., M, 3]
        :param width: The width of transitions. The shape is [..., M]
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param B: The magnetic fields where spectra should be created.
        :return: result: Tensor of shape (..., N) with the value of the integral for each B
        """
        B = B[:10]
        width = self.eps_val + width * constants.PLANCK / constants.BOHR
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=False)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)

        d1 = B1 - B3
        d2 = B2 - B3

        denom1 = (d1 - d2) + self.eps_val
        denom2 = d1 + self.eps_val
        d2_safe = (d2 + self.eps_val)

        c = self.two_sqrt / width
        c_exp = c.unsqueeze(-1)
        width_exp = width.unsqueeze(-1)
        def F(x):
            return x * torch.erf(c_exp * x) + (width_exp / (self.two_sqrt * self.sqrt_pi)) * torch.exp(
                - (c_exp * x) ** 2)


        X1 = F(B1.unsqueeze(-1) - B)
        X2 = F(B2.unsqueeze(-1) - B)
        X3 = F(B3.unsqueeze(-1) - B)

        term1 = (X1 - X2) / denom1.unsqueeze(-1)
        term2 = (X1 - X3) / denom2.unsqueeze(-1)
        term = term1 - term2
        factor = A_mean / d2_safe
        print(term * factor.unsqueeze(-1))
        print((term1 < 0).sum())
        print((term2 < 0).sum())
        result = torch.einsum('...mn,m->...n', term * factor.unsqueeze(-1), area)
        return result




