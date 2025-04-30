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


def get_induced_speed_matrix(speed_data: tuple[int, int, torch.Tensor], size: int = 4):
    idx_1, idx_2, speed = speed_data
    induced_speed = torch.zeros((size, size))
    induced_speed[idx_1, idx_2] = speed
    induced_speed[idx_1, idx_1] = -speed
    induced_speed[idx_2, idx_1] = speed
    induced_speed[idx_2, idx_2] = -speed
    return induced_speed

class EvolutionMatrix:
    def __init__(self, res_energies: torch.Tensor):
        """
        :param res_energies: The resonance energies. The shape is [..., spin system dimension, spin system dimension]
        """
        self.energy_diff = res_energies.unsqueeze(-2) - res_energies.unsqueeze(-1)    # Think about it!!!!
        self.energy_diff = constants.unit_converter(self.energy_diff)
        self.config_dim = self.energy_diff.shape[:-2]

    def _get_energy_factor(self, temp: torch.Tensor):
        denom = 1 + torch.exp(-self.energy_diff / temp.view(-1, *[1] * (self.energy_diff.dim())))   # Must be speed up via batching of temperature
        return torch.reciprocal(denom)

    def __call__(self, temp: torch.Tensor, free_speed: RelaxationSpeedMatrix, induced_speed: torch.Tensor):
        """
        :param temp: The time-dependant temperature. The shape is [time_size]
        :param free_speed: The free relaxation speed. The shape of the __call__ is
        :param induced_speed:
        :return:
        """
        temp = temp
        energy_factor = self._get_energy_factor(temp)
        part = 2 * free_speed(temp) * energy_factor
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

def get_relaxation_speeds(additional_config=1) -> RelaxationSpeedMatrix:
    #amplitude_relaxation_time_triplet = 100_000  # us
    #amplitude_relaxation_time_exchange = 500  # us

    amplitude_relaxation_time_triplet = torch.tensor(5_700)
    amplitude_relaxation_time_triplet =\
        amplitude_relaxation_time_triplet.view(-1, *[1] * additional_config)

    amplitude_relaxation_time_exchange = torch.tensor(500)
    amplitude_relaxation_time_exchange = \
        amplitude_relaxation_time_exchange.view(-1, *[1] * additional_config)

    zero_speed = torch.tensor(torch.inf)
    zero_speed = \
        zero_speed.view(-1, *[1] * additional_config)

    #delta_temp = 8.05
    delta_temp = 4.05
    #zero_temp = 5.0
    zero_temp = 5.0
    triplet_speed = RelaxationSpeedExponential(amplitude_relaxation_time_triplet, zero_temp, 3*delta_temp)
    exchange_speed = RelaxationSpeedExponential(amplitude_relaxation_time_exchange, zero_temp, 3*delta_temp)
    zero_speed = RelaxationSpeedExponential(zero_speed)

    #free_speeds = [[zero_speed, triplet_speed, zero_speed, zero_speed],
    #               [triplet_speed, zero_speed, exchange_speed, triplet_speed],
    #               [zero_speed, exchange_speed, zero_speed, zero_speed],
    #               [zero_speed, triplet_speed, zero_speed, zero_speed],
    #               ]


    #free_speeds = [[zero_speed, zero_speed, triplet_speed, zero_speed],
    #               [zero_speed, zero_speed, exchange_speed, zero_speed],
    #               [triplet_speed, exchange_speed, zero_speed, triplet_speed],
    #               [zero_speed, zero_speed, triplet_speed, zero_speed],
    #               ]

    free_speeds = [[zero_speed, zero_speed, exchange_speed, zero_speed],
                   [zero_speed, zero_speed, triplet_speed, zero_speed],
                   [exchange_speed, triplet_speed, zero_speed, triplet_speed],
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

    def _precompute_data(self, res_fields, vector_down, vector_up, energies, lvl_down, lvl_up):
        energies = copy.deepcopy(energies)
        return res_fields, vector_down, vector_up, energies, lvl_down, lvl_up

    def _get_initial_populations(self, energies):
        return nn.functional.softmax(-constants.unit_converter(energies) / self.start_temp, dim=-1)

    def _matrix_and_profile(self, energies: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor):
        free = get_relaxation_speeds()
        ind = get_induced_speed(lvl_down, lvl_up).unsqueeze(0)
        evo = EvolutionMatrix(energies)
        prof = TempProfile(power=50)
        return free, ind, evo, prof

    def _post_compute_data(self, initial_populations, time_dep_population, lvl_down, lvl_up):
        indexes = torch.arange(initial_populations.shape[-2], device=initial_populations.device)
        time_intensities = time_dep_population[..., indexes, lvl_down] - time_dep_population[..., indexes, lvl_up]
        init_intensity = initial_populations[..., indexes, lvl_down] - initial_populations[..., indexes, lvl_up]
        return time_intensities - init_intensity.unsqueeze(0)

    def run(self, time: torch.Tensor, initial_populations: torch.Tensor, prof: TempProfile, evo: EvolutionMatrix,
            free: RelaxationSpeedMatrix, ind: torch.Tensor):
        dt = (time[1] - time[0]) * 1e-6
        size = time.size()[0]
        temp = prof(time)
        n = torch.zeros((size,) + initial_populations.shape, dtype=initial_populations.dtype)
        n[0] = initial_populations
        M = evo(temp, free, ind)
        exp_M = torch.matrix_exp(M * dt)

        """
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
        """
        for i in range(len(time) - 1):

            current_n = n[i]  # Shape [..., K]
            next_n = torch.einsum('...kl,...l->...k', exp_M[i], current_n)
            #delta_n = torch.einsum('...kl,...l->...k', M_i, current_n)
            #print(delta_n[0])
            #next_n = delta_n * dt + current_n
            #next_n = next_n / next_n.sum(dim=-1, keepdim=True)  # To make sure that it is equel to 1
            n[i + 1] = next_n
        return n

    def __call__(self, res_fields, vector_down, vector_up, energies, lvl_down, lvl_up, *args, **kwargs):
        """
        :param energies: energies in Hz
        :return: population_differences
        """
        eigen_vectors_full = args[0]
        eigen_vectors_base = 


        res_fields, vector_down, vector_up, energies, lvl_down, lvl_up = self._precompute_data(res_fields, vector_down,
                                                                                               vector_up, energies,
                                                                                               lvl_down, lvl_up
                                                                                               )
        initial_populations = self._get_initial_populations(energies)
        free, ind, evo, prof = self._matrix_and_profile(energies, lvl_down, lvl_up)
        size = 3000
        time = torch.linspace(-100, 70 * 1e3, size)
        n = self.run(time, initial_populations, prof, evo, free, ind)
        res = self._post_compute_data(initial_populations, n, lvl_down, lvl_up)
        return res

"""
    def rate_equation(self, t, n_flat, multiplication_matrix, free_speed, induced_speed, temp_profile):

        RHS for dn/dt = M(t) n, where M depends on t through temperature.
        - t: scalar time
        - n_flat: flattened populations of shape (..., K)
        Returns dn_flat/dt of same shape.

        print(t)
        #K = multiplication_matrix.energies.shape[-1]
        #n = n_flat.view(*multiplication_matrix.energies.shape[:-1], K)
        temp = temp_profile(t)
        M_t = multiplication_matrix(temp, free_speed, induced_speed)

        # Compute dn/dt = M_t @ n
        dn = torch.einsum("...kl,...l->...k", M_t, n_flat)
        return dn
"""