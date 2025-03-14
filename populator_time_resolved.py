from spectral_integration import BaseSpectraIntegrator, SpectraIntegratorExtended, SpectraIntegratorEasySpinLike

import torch
from torch import nn

import constants
import mesher
import res_field_algorithm
import spin_system

import typing as tp


class RelaxationSpeed:
    def __init__(self, time_amplitude=None, zero_temp=3.5, delta_temp=3.5):
        if time_amplitude is None:
            self.speed_amplitude = 0.0
        else:
            self.speed_amplitude = 1 / (2 * time_amplitude * 10**(-6))
        self.zero_temp = zero_temp
        self.delta_temp = delta_temp

    def __call__(self, temp):
        return torch.exp((temp - self.zero_temp) / self.delta_temp) * self.speed_amplitude

    def __eq__(self, other):
        return (self.speed_amplitude == other.speed_amplitude) and\
            (self.zero_temp == other.zero_temp) and\
            (self.delta_temp == other.delta_temp)


class MultiplicationMatrix:
    def __init__(self, levels: tp.Collection[EnergyLevel],
                 free_speeds: list[list[RelaxationSpeed]], induced_speeds: torch.Tensor | None | list = None):
        if induced_speeds is None:
            self.induced_speeds = torch.zeros((len(levels), len(levels)))
        else:
            self.induced_speeds = torch.tensor(induced_speeds)
        self.levels = levels
        self.free_speeds = free_speeds
        self._diagonal_indexes = torch.diag_indices(len(levels))
        lines = range(len(free_speeds))
        columns = range(len(free_speeds[0]))
        specific_speed = torch.tensor([[free_speeds[i][j](3.5) for i in lines] for j in columns])

        if self.induced_speeds.shape != specific_speed.shape:
            raise ValueError("induced_speeds and free_speeds must have the same shapes")

        if (specific_speed.shape[0] != specific_speed.shape[1]) or (specific_speed.shape[0] != len(levels)):
            raise ValueError("speeds must have the length n x n, where n is len of the levels array")

        if (specific_speed != specific_speed.T).any():
            raise ValueError("free_speeds must be symetric matrix")

        if (self.induced_speeds != self.induced_speeds.T).any():
            raise ValueError("induced_speeds must be symetric matrix")

        if (torch.diagonal(specific_speed) != 0.0).any():
            raise ValueError("free_speeds must have zero diagonal elements")

        if (torch.diagonal(self.induced_speeds) != 0.0).any():
            raise ValueError("induced_speeds must have zero diagonal elements")

    def __call__(self, temp):
        size = len(self.levels)
        temp_factors = torch.tensor([[1 / (1 + (level_1 / level_2)(temp))
                                  for level_1 in self.levels] for level_2 in self.levels])
        specific_speed = torch.array([[self.free_speeds[i][j](temp) + self.induced_speeds[i][j] if i != j else 0.0 for
                                    i in range(size)] for j in range(size)])
        transition_matrix = 2 * specific_speed * temp_factors
        transition_matrix[self._diagonal_indexes] = -torch.sum(transition_matrix, axis=0)
        return transition_matrix

class TempProfile:
    def __init__(self, power=50,
                 impulse_width=80, thermal_relax_width=2 * 10**4,
                 start_temp=3.5, heat_capacity=0.257 * 3 * 10**-4):
        self.power = torch.tensor(power)  # in Wt
        self.impulse_width = torch.tensor(impulse_width)  # in us
        self.thermal_relax_width = torch.tensor(thermal_relax_width)  # in us
        self.start_temp = torch.tensor(start_temp)  # in K
        self.heat_capacity = torch.tensor(heat_capacity)  # in J / K

    def __call__(self, time):
        delta_T_factor = self.power * self.impulse_width * 10**(-6) * 4 * self.start_temp**3 / (2 * self.heat_capacity)
        max_temp = (2 * delta_T_factor + self.start_temp**4) ** 0.25
        if time < self.impulse_width:
            temp = (delta_T_factor * (torch.sin((time / self.impulse_width - 0.5) * torch.pi) + 1.0) +
                    self.start_temp**4
                    ) ** 0.25 - \
                   (max_temp - self.start_temp) * (1 - torch.exp(-time / self.thermal_relax_width))
        else:
            temp = max_temp - (max_temp - self.start_temp) * (1 - torch.exp(-time / self.thermal_relax_width))
        return temp

class PopulatorTimeResolved:
    def __init__(self, start_temp: float = 3.5):
        """
        :param start_temp: temperature in K
        """

        self.start_temp = start_temp

    def population_matrix(self):

    def __call__(self, res_fields, vector_down, vector_up, energies, mask_triu, lvl_down, lvl_up):
        """
        :param energies: energies in Hz
        :return: population_differences
        """
        print(mask_triu)
        initial_populations = nn.functional.softmax(-constants.unit_converter(energies) / self.start_temp, dim=-1)
        indexes = torch.arange(initial_populations.shape[-2], device=initial_populations.device)
        return initial_populations[..., indexes, lvl_up] - initial_populations[..., indexes, lvl_down]