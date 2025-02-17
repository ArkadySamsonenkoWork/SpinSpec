import warnings

import torch
from torch import nn

import constants


class StationaryPopulator:
    def __init__(self):
        pass

    def __call__(self, energies, temperature, u, v):
        """
        :param energies: energies in Hz
        :param temperature: temperature in K
        :return: population_differences
        """
        populations = nn.functional.softmax(-constants.unit_converter(energies) / temperature, dim=-1)
        populations[u] - populations[v]
        return populations[u] - populations[v]

class Broadening:
    def __init__(self):
        pass

    def __call__(self, ):
        pass


class IntensitiesCalculator:
    def __init__(self):
        self.tolerancy = 1e-9
        self.threshold = 1e-4

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
        return 1 / (factor_1 - factor_2 + self.tolerancy)

    def __call__(self, Gx, Gy, Gz, batches):
        for batch in batches:
            indexes = batch["indexes"]
            vector_down, vector_up = batch["vectors"]
            mask_cut = batch["mask_cut"]
            intensity = mask_cut * (
                    self._compute_magnitization(Gx, Gy, vector_down, vector_up, indexes) +
                    self._freq_to_field(vector_down, vector_up, Gz, indexes)
            )
            resonance_energies = batch["energies"]
            batch[intensity] = intensity


class Mesh:
    def __init__(self):
        pass

    def interpolate(self):
        pass

    def get_init_mesh(self):
        pass

class SpectraCreator():
    def __init__(self, spin_system_dim, batch_dims):
        self.spin_system_dim = spin_system_dim
        self.batch_dims = batch_dims
    def combine_to_one_batch(self, batches):
        num_pairs = (self.spin_system_dim ** 2 - self.spin_system_dim) // 2
        res_fields = torch.zeros((*self.batch_dims, num_pairs), dtype=torch.float32)
        intensities = torch.zeros((*self.batch_dims, num_pairs), dtype=torch.float32)
        width = torch.zeros((*self.batch_dims, num_pairs), dtype=torch.float32)
        for batch in batches:
            batch_mask_cut = batch["mask_cut"]
            batch_indexes = batch["indexes"]
            batch_intensities = batch["intensities"]
            batch_width = batch["width"]
            batch_res_fields = batch["B_res"]
            
            res_fields[batch_indexes][batch_mask_cut] = batch_res_fields
            intensities[batch_indexes][batch_mask_cut] = batch_intensities
            width[batch_indexes][batch_mask_cut] = batch_width











