import warnings
import math

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

class SpectraIntegrator:
    def __init__(self):
        self.sqrt_pi = torch.tensor(math.sqrt(math.pi))
        self.eps_val = torch.tensor(1e-12)

    def _compute_free_factor(self, B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor, alpha: torch.Tensor):
        # Store the fixed parameters (scalars or 0-dim tensors)
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        self.alpha = alpha

        self.p = B1 - B3
        self.q = B2 - B3
        # sqrt(alpha) is used repeatedly
        self.sqrt_alpha = torch.sqrt(alpha)

        eps = self.eps_val * torch.ones_like(self.p)
        self.p_safe = torch.where(torch.abs(self.p) < eps, eps, self.p)
        self.q_safe = torch.where(torch.abs(self.q) < eps, eps, self.q)
        self.pq_safe = torch.where(torch.abs(self.p - self.q) < eps, eps, self.p - self.q)

        return (alpha * self.sqrt_pi) / (2 * self.q_safe)

    def integrate(self, B1, B2, B3, B, alpha):
        r"""Analytically integrates the function

             f(l1,l2) = exp( -((l1*B1 + l2*B2 + (1-l1-l2)*B3 - B)^2) / alpha )

        over the reference triangle in barycentric coordinates (l1>=0, l2>=0, l1+l2<=1).

        The derivation uses the substitutions:
          p = B1 - B3,  q = B2 - B3,  d = B3 - B,
        and the closed-form expression (in terms of error functions) is given by

             I_bary = (alpha*sqrt(pi))/(2*q) * { 1/(p-q) * [F((d+p)/sqrt(alpha)) - F((d+q)/sqrt(alpha))]
                                              - 1/p * [F((d+p)/sqrt(alpha)) - F(d/sqrt(alpha))] },
        with F(u)= u*erf(u) + exp(-u^2)/sqrt(pi).

        Parameters:
           B1, B2, B3: function Bc evaluated at triangle vertices (scalars or tensors)
           B         : the scalar B (can be a tensor for batch evaluation)
           alpha     : mean value of alpha over the triangle

        Returns:
           I_bary    : the value of the barycentric integral.
        """
        d = self.B3 - B

        # Compute the u values that depend on d
        u1 = d / self.sqrt_alpha
        u2 = (d + self.q) / self.sqrt_alpha
        u3 = (d + self.p) / self.sqrt_alpha

        # Inline the function F(u) = u * erf(u) + exp(-u^2)/sqrt(pi)
        F_u1 = u1 * torch.erf(u1) + torch.exp(-u1 * u1) / self.sqrt_pi
        F_u2 = u2 * torch.erf(u2) + torch.exp(-u2 * u2) / self.sqrt_pi
        F_u3 = u3 * torch.erf(u3) + torch.exp(-u3 * u3) / self.sqrt_pi

        # Compute the two terms in the expression (vectorized over B)
        term1 = (F_u3 - F_u2) / self.pq_safe
        term2 = (F_u3 - F_u1) / self.p_safe
        factor = self._compute_free_factor(B1, B2, B3, alpha)

        return factor * (term1 - term2)




