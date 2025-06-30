from spectral_integration import BaseSpectraIntegrator, SpectraIntegratorExtended, SpectraIntegratorEasySpinLike
from abc import ABC, abstractmethod
import torch
from torch import nn
from torchdiffeq import odeint


import constants
import mesher
import res_field_algorithm
import spin_system

import copy
import typing as tp

from . import transform


class EvolutionMatrix:
    def __init__(self, res_energies: torch.Tensor, symmetry_probs: bool = True):
        """
        :param res_energies: The resonance energies. The shape is [..., spin system dimension, spin system dimension]
        """
        self.energy_diff = res_energies.unsqueeze(-2) - res_energies.unsqueeze(-1)  # Think about it!!!!
        self.energy_diff = constants.unit_converter(self.energy_diff)
        self.config_dim = self.energy_diff.shape[:-2]
        self._probs_matrix = self._prob_matrix_factory(symmetry_probs)

    def _prob_matrix_factory(self, symmetry_probs: bool):
        if symmetry_probs:
            return self._compute_boltzmann_symmetry
        else:
            return self._compute_boltzmann_complement

    def _compute_energy_factor(self, temp: torch.Tensor):
        denom = 1 + torch.exp(-self.energy_diff / temp)  # Must be speed up via batching of temperature
        return torch.reciprocal(denom)

    def _compute_boltzmann_symmetry(self, temp: torch.tensor, free_probs: torch.Tensor):
        energy_factor = self._compute_energy_factor(temp)
        probs_matrix = 2 * free_probs * energy_factor
        return probs_matrix

    def _compute_boltzmann_complement(self, temp: torch.tensor, free_probs: torch.Tensor):
        numerator = torch.exp(self.energy_diff / temp)
        probs_matrix = torch.where(free_probs == 0, free_probs.transpose(-1, -2) * numerator, free_probs)
        return probs_matrix

    def __call__(self, temp: torch.tensor,
                 free_probs: torch.Tensor,
                 induced_probs: torch.Tensor | None = None,
                 out_probs: torch.Tensor | None = None):
        """
        :temp
        :param free_probs: The free relaxation speed. The shape of the __call__ is
        :param induced_probs:
        :param out_probs:
        :return:
        """
        probs_matrix = self._probs_matrix(temp, free_probs)
        K = probs_matrix.shape[-1]
        indices = torch.arange(K, device=probs_matrix.device)
        probs_matrix[..., indices, indices] = -probs_matrix.sum(dim=-2)
        transition_matrix = probs_matrix

        if induced_probs is not None:
            induced_probs[..., indices, indices] = -induced_probs.sum(dim=-2)
            transition_matrix += induced_probs
        if out_probs is not None:
            transition_matrix -= torch.diag_embed(out_probs)
        return transition_matrix


class TransitionMatrixGenerator(ABC):
    def __init__(self, context: dict[str, tp.Any] | None = None, *args, **kwargs):
        self.context = context or {}

    def __call__(self, time: torch.Tensor) ->\
            tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        :param time: time, where transition matrix should be found
        :return: tuple [temperature, base probabilities of transition,
        induced probabilities, outgoing probabilities]
        """
        temp = self._temperature(time)
        base_probs = self._base_transition_probs(temp)
        induced = self._induced_transition_probs(temp)
        outgoing = self._outgoing_transition_probs(temp)
        return temp, base_probs, induced, outgoing

    def _temperature(self, time: torch.Tensor) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return None

    @abstractmethod
    def _base_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor:
        """Return the free transition probabilities at given temperature(s)"""
        pass

    def _induced_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor | None:
        """Optional induced transitions; default None"""
        return None

    def _outgoing_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor | None:
        """Optional outgoing transitions; default None"""
        return None


class EvolutionSolver:
    @staticmethod
    def odeint_solver(time: torch.Tensor, initial_populations: torch.Tensor,
                     evo: EvolutionMatrix, matrix_generator: TransitionMatrixGenerator):
        def _rate_equation(t, n_flat, evo: EvolutionMatrix, matrix_generator: TransitionMatrixGenerator):
            """
            RHS for dn/dt = M(t) n, where M depends on t through temperature.
            - t: scalar time
            - n_flat: flattened populations of shape (..., K)
            Returns dn_flat/dt of same shape.
            """
            M_t = evo(*matrix_generator(t))
            dn = torch.matmul(M_t, n_flat.unsqueeze(-1)).squeeze(-1)
            return dn
        sol = odeint(func=lambda t, y: _rate_equation(
                     t, y, evo, matrix_generator),
                     y0=initial_populations,
                     t=time
                     )
        return sol

    @staticmethod
    def exponential_solver(time: torch.Tensor,
                          initial_populations: torch.Tensor,
                              evo: EvolutionMatrix, matrix_generator: TransitionMatrixGenerator):
        dt = (time[1] - time[0])
        M = evo(*matrix_generator(time))
        exp_M = torch.matrix_exp(M * dt)

        size = time.size()[0]
        n = torch.zeros((size,) + initial_populations.shape, dtype=initial_populations.dtype)
        n[0] = initial_populations

        for i in range(len(time) - 1):
            current_n = n[i]  # Shape [..., K]
            next_n = torch.matmul(exp_M[i], current_n.unsqueeze(-1)).squeeze(-1)
            n[i + 1] = next_n
        return n

    @staticmethod
    def stationary_rate_solver(time: torch.Tensor,
                         initial_populations: torch.Tensor,
                         evo: EvolutionMatrix, matrix_generator: TransitionMatrixGenerator):
        M = evo(*matrix_generator(time[0]))

        dims_to_add = M.dim()

        reshape_dims = [len(time)] + [1] * dims_to_add
        time_reshaped = time.reshape(reshape_dims)

        exp_m = torch.matrix_exp(M * time_reshaped)
        n = torch.matmul(exp_m, initial_populations.unsqueeze(-1)).squeeze(-1)
        return n




