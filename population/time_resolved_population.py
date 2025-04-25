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


class RelaxationSpeedExponential:
    def __init__(self, time_amplitude=None, zero_temp=3.5, delta_temp=3.5):
        if time_amplitude is None:
            self.speed_amplitude = 0.0
        else:
            self.speed_amplitude = 1 / (2 * time_amplitude * 10**(-6))
        self.zero_temp = zero_temp
        self.delta_temp = delta_temp

    def __call__(self, temp):
        return torch.exp((temp - self.zero_temp) / self.delta_temp) * self.speed_amplitude

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

    def __call__(self, temp):
        #speed_amplitudes = torch.tensor([[m.speed_amplitude for m in row] for row in self.relaxation_speeds])
        #zero_temps = torch.tensor([[m.zero_temp for m in row] for row in self.relaxation_speeds])
       # delta_temps = torch.tensor([[m.delta_temp for m in row] for row in self.relaxation_speeds])
        result = torch.tensor([[m(temp) for m in row] for row in self.relaxation_speeds])
        return result


def get_induced_speed_matrix(speed_data: tuple[int, int, torch.Tensor], size: int = 4):
    idx_1, idx_2, speed = speed_data
    induced_speed = torch.zeros((size, size))
    induced_speed[idx_1, idx_2] = speed
    induced_speed[idx_1, idx_1] = -speed
    induced_speed[idx_2, idx_1] = speed
    induced_speed[idx_2, idx_2] = -speed
    return induced_speed

class MultiplicationMatrix:
    def __init__(self, res_energies: torch.Tensor):
        self.energy_diff = res_energies.unsqueeze(-2) - res_energies.unsqueeze(-1)    # Think about it!!!!
        self.energy_diff = constants.unit_converter(self.energy_diff)

    def __call__(self, temp: torch.Tensor, free_speed: RelaxationSpeedMatrix, induced_speed: torch.Tensor):
        """
        :param temp:
        :param free_speed:
        :param induced_speed:
        :return:
        """

        denom = 1 + torch.exp(-self.energy_diff / temp)  # Must be speed up via batching of temperature
        part = 2 * free_speed(temp) / denom
        K = part.size(-1)
        indices = torch.arange(K, device=part.device)
        part[..., indices, indices] = -part.sum(dim=-2)
        transition_matrix = part + induced_speed
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

def get_relaxation_speeds():
    #amplitude_relaxation_time_triplet = 100_000  # us
    #amplitude_relaxation_time_exchange = 500  # us

    amplitude_relaxation_time_triplet = 5_700
    amplitude_relaxation_time_exchange = 500

    #delta_temp = 8.05
    delta_temp = 4.05
    #zero_temp = 5.0
    zero_temp = 5.0
    triplet_speed = RelaxationSpeedExponential(amplitude_relaxation_time_triplet, zero_temp, 3*delta_temp)
    exchange_speed = RelaxationSpeedExponential(amplitude_relaxation_time_exchange, zero_temp, 3*delta_temp)
    zero_speed = RelaxationSpeedExponential()

    #free_speeds = [[zero_speed, triplet_speed, zero_speed, zero_speed],
    #               [triplet_speed, zero_speed, exchange_speed, triplet_speed],
    #               [zero_speed, exchange_speed, zero_speed, zero_speed],
    #               [zero_speed, triplet_speed, zero_speed, zero_speed],
    #               ]


    free_speeds = [[zero_speed, zero_speed, triplet_speed, zero_speed],
                   [zero_speed, zero_speed, exchange_speed, zero_speed],
                   [triplet_speed, exchange_speed, zero_speed, triplet_speed],
                   [zero_speed, zero_speed, triplet_speed, zero_speed],
                   ]
    return RelaxationSpeedMatrix(free_speeds)

def get_induced_speed(lvl_down, lvl_up):
    amplitude_relaxation_time_triplet = 3_000
    induced_amplitude = 1 / (2 * amplitude_relaxation_time_triplet * 10**(-6))
    induced_amplitude = induced_amplitude / 1000
    speed_data = (lvl_down[0], lvl_up[0], induced_amplitude)
    speed_matrix_1 = get_induced_speed_matrix(speed_data)  # Must be rebuild
    speed_data = (lvl_down[1], lvl_up[1], induced_amplitude)
    speed_matrix_2 = get_induced_speed_matrix(speed_data)
    indeuced_matrix = torch.stack((speed_matrix_1, speed_matrix_2), dim=0)
    return indeuced_matrix


class TimeResolvedPopulator:
    def __init__(self, start_temp: float = 3.5):
        """
        :param start_temp: temperature in K
        """
        self.start_temp = start_temp


    def __call__(self, res_fields, vector_down, vector_up, energies, lvl_down, lvl_up):
        """
        :param energies: energies in Hz
        :return: population_differences
        """
        energies = copy.deepcopy(energies)
        energies[..., 1] = energies[..., 1] * 7
        #energies[..., 2] = energies[..., 2] * 12

        initial_populations = nn.functional.softmax(-constants.unit_converter(energies) / self.start_temp, dim=-1)
        indexes = torch.arange(initial_populations.shape[-2], device=initial_populations.device)
        initial_population_intensity =\
            initial_populations[..., indexes, lvl_down] - initial_populations[..., indexes, lvl_up]
        free_speed_matrix = get_relaxation_speeds()
        induced_speed = get_induced_speed(lvl_down, lvl_up).unsqueeze(0)
        multiplication_matrix = MultiplicationMatrix(energies)
        temp_profiler = TempProfile(power=50)
        size = 3000
        #size = 200
        time = torch.linspace(-100, 70 * 1e3, size)
        n0 = initial_populations

        dt = (time[1] - time[0]) * 1e-6
        temp = temp_profiler(time)

        n = torch.zeros((size,) + n0.shape, dtype=n0.dtype)
        n[0] = n0
        # Iterate over each time step to compute the solution
        #M = multiplication_matrix(temp, free_speed_matrix, induced_speed)  # Shape [..., K, K]
        #exp_M = torch.matrix_exp(M * dt)

        self.i = 0
        sol = odeint(
            func=lambda t, y: self.rate_equation(
                t, y, multiplication_matrix, free_speed_matrix, induced_speed, temp_profiler),
            y0=n0,
            t=time,
            rtol=1e-2,
            atol=1e-0,
            #method=ode_solver
        )
        
        print(sol.shape)
        for i in range(len(time) - 1):

            current_n = n[i]  # Shape [..., K]
            next_n = torch.einsum('...kl,...l->...k', exp_M, current_n)
            #delta_n = torch.einsum('...kl,...l->...k', M_i, current_n)
            #print(delta_n[0])
            #next_n = delta_n * dt + current_n
            #next_n = next_n / next_n.sum(dim=-1, keepdim=True)  # To make sure that it is equel to 1
            n[i + 1] = next_n

        intensities = n[..., indexes, lvl_down] - n[..., indexes, lvl_up]

        return intensities

    def rate_equation(self, t, n_flat, multiplication_matrix, free_speed, induced_speed, temp_profile):
        """


        RHS for dn/dt = M(t) n, where M depends on t through temperature.
        - t: scalar time
        - n_flat: flattened populations of shape (..., K)
        Returns dn_flat/dt of same shape.
        """
        print(t)
        #K = multiplication_matrix.energies.shape[-1]
        #n = n_flat.view(*multiplication_matrix.energies.shape[:-1], K)
        temp = temp_profile(t)
        M_t = multiplication_matrix(temp, free_speed, induced_speed)

        # Compute dn/dt = M_t @ n
        dn = torch.einsum("...kl,...l->...k", M_t, n_flat)

        return dn