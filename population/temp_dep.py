from spectral_integration import BaseSpectraIntegrator, SpectraIntegratorExtended, SpectraIntegratorEasySpinLike

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

class RelaxationSpeedExponential:
    def __init__(self, time_amplitude: torch.Tensor, zero_temp=3.5, delta_temp=3.5):
        """
        :param time_amplitude: the relaxation time. The shape is [...]
        :param zero_temp: offset of the exponential dependence
        :param delta_temp: the scale of the exponential dependence
        """
        self.speed_amplitude = 1 / (2 * time_amplitude * 10**(-6))
        self.zero_temp = zero_temp
        self.delta_temp = delta_temp

    def __call__(self, temp: torch.Tensor):
        """
        :param temp: time-dependant torch tensor. The shape is [time dimension]
        :return:
        """
        out =\
            torch.exp((temp.view(-1, *[1] * (self.speed_amplitude.dim())) - self.zero_temp) / self.delta_temp) *\
            self.speed_amplitude
        return out

    def __neg__(self):
        speed = RelaxationSpeedExponential(zero_temp=self.zero_temp, delta_temp=self.delta_temp)
        speed.speed_amplitude = -self.speed_amplitude
        return speed

    def __sub__(self, other):
        if (other.zero_temp == self.zero_temp) and (other.delta_temp == self.delta_temp):
            speed = RelaxationSpeedExponential(zero_temp=self.zero_temp, delta_temp=self.delta_temp)
            speed.speed_amplitude = self.speed_amplitude - other.speed_amplitude
        else:
            speed = RelaxationSpeedBiExponential(zero_temps=(self.zero_temp, other.zero_temp),
                                                 delta_temps=(self.delta_temp, other.delta_temp))
            speed.speed_amplitudes = (self.speed_amplitude, -other.speed_amplitude)
        return speed

    def __eq__(self, other):
        return (self.speed_amplitude == other.speed_amplitude) and\
            (self.zero_temp == other.zero_temp) and\
            (self.delta_temp == other.delta_temp)

class RelaxationSpeedBiExponential(RelaxationSpeedExponential):
    def __init__(self, time_amplitudes=None, zero_temps=(3.5, 3.5), delta_temps=(3.5, 3.5)):
        if time_amplitudes is None:
            self.speed_amplitudes = (0.0, 0.0)
        else:
            self.speed_amplitudes = (1 / (2 * time_amplitudes[0] * 10**(-6)), 1 / (2 * time_amplitudes[1] * 10**(-6)))
        self.zero_temps = zero_temps
        self.delta_temps = delta_temps

    def __call__(self, temp):
        return torch.exp((temp - self.zero_temp) / self.delta_temp) * self.speed_amplitudes

    def __neg__(self):
        speed = RelaxationSpeedBiExponential(zero_temps=self.zero_temps, delta_temps=self.delta_temps)
        speed.speed_amplitudes = (-self.speed_amplitudes[0], -self.speed_amplitudes[1])
        return speed

    def __sub__(self, other):
        speed = RelaxationSpeedExponential(zero_temp=self.zero_temp, delta_temp=self.delta_temp)
        speed.speed_amplitudes = -self.speed_amplitude
        return speed

    def __eq__(self, other):
        return (self.speed_amplitudes == other.speed_amplitude) and\
            (self.zero_temps == other.zero_temps) and\
            (self.delta_temps == other.delta_temps)


class RelaxationSpeedMatrix:
    def __init__(self, relaxation_speeds: list[list[RelaxationSpeedExponential]]):
        self.relaxation_speeds = relaxation_speeds

    def __call__(self, temp: torch.Tensor) -> torch.Tensor:
        """
        :param temp: The time-dependant temperature. The shape is [batch_size, time_size]
        :return: relaxation_spin:
        relaxation matrix at specific temperature. The output shape is [..., spin dimension, spin dimension]
        """
        result = [[m(temp) for m in row] for row in self.relaxation_speeds]
        result = torch.stack([torch.stack(row, dim=-1) for row in result], dim=-2)
        return result


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
        """
        :param time: in us
        :return:
        """
        delta_T_factor = self.power * self.impulse_width * 10**(-6) * 4 * self.start_temp**3 / (2 * self.heat_capacity)
        max_temp = (2 * delta_T_factor + self.start_temp**4) ** 0.25
        sigmoid_part = (1 - self.start_temp / max_temp) * torch.nn.functional.sigmoid(time / self.impulse_width - 4) +\
                       self.start_temp / max_temp
        relaxation_part = max_temp - (max_temp - self.start_temp) * (1 - torch.exp(-time / self.thermal_relax_width))
        return relaxation_part * sigmoid_part