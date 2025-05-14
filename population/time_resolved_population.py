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


class BaseTimeResolvedPopulator:
    def __init__(self, start_temp: float = 300, config: dict[str, tp.Any] | None = None):
        """
        :param start_temp: temperature in K
        """
        self.start_temp = torch.tensor(start_temp)
        self.config = config

    def __call__(self, res_fields, vector_down, vector_up, energies, lvl_down, lvl_up, *args, **kwargs):
        pass


class TimeResolvedPopulator(BaseTimeResolvedPopulator):
    def _precompute_data(self, res_fields, vector_down, vector_up, energies, lvl_down, lvl_up):
        energies = copy.deepcopy(energies)
        return res_fields, vector_down, vector_up, energies, lvl_down, lvl_up

    def _get_initial_populations(self, energies):
        return nn.functional.softmax(-constants.unit_converter(energies) / self.start_temp, dim=-1)

    def _odeint_solve(self, time: torch.Tensor, initial_populations: torch.Tensor, prof: TempProfile,
                      evo: EvolutionMatrix, free: RelaxationSpeedMatrix, ind: torch.Tensor):
        sol = odeint(
            func=lambda t, y: self._rate_equation(
                t, y, evo, free, ind, prof),
            y0=initial_populations,
            t=time,
        )
        return sol

    def _exponential_solve(self, time: torch.Tensor, initial_populations: torch.Tensor,
                           prof: TempProfile, evo: EvolutionMatrix, free: RelaxationSpeedMatrix, ind: torch.Tensor):
        temp = prof(time)
        dt = (time[1] - time[0])
        M = evo(temp, free, ind)
        exp_M = torch.matrix_exp(M * dt)

        size = time.size()[0]
        n = torch.zeros((size,) + initial_populations.shape, dtype=initial_populations.dtype)
        n[0] = initial_populations

        for i in range(len(time) - 1):
            current_n = n[i]  # Shape [..., K]
            next_n = torch.matmul(exp_M[i], current_n.unsqueeze(-1)).squeeze(-1)
            n[i + 1] = next_n
        return n

    def _rate_equation(self, t, n_flat, multiplication_matrix, free_speed, induced_speed, temp_profile):
        """
        RHS for dn/dt = M(t) n, where M depends on t through temperature.
        - t: scalar time
        - n_flat: flattened populations of shape (..., K)
        Returns dn_flat/dt of same shape.
        """

        temp = temp_profile(t)
        M_t = multiplication_matrix(temp, free_speed, induced_speed)

        # Compute dn/dt = M_t @ n
        dn = torch.matmul(M_t, n_flat.unsqueeze(-1)).squeeze(-1)
        return dn


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
        return self._exponential_solve(time, initial_populations, prof, evo, free, ind)

        #return self._odeint_solve(time, initial_populations, prof, evo, free, ind)

    def __call__(self, res_fields, vector_down, vector_up, energies, lvl_down, lvl_up, *args, **kwargs):
        """
        :param energies: energies in Hz
        :return: population_differences
        """
        eigen_vectors_full = args[0]
        eigen_vectors_base = torch.eye(4, 4, dtype=torch.complex64)
        eigen_vectors_base = eigen_vectors_base.unsqueeze(0)
        eigen_vectors_base = eigen_vectors_base.unsqueeze(0)
        answer = transform.basis_transformation(eigen_vectors_base, eigen_vectors_full)
        print(answer[0, 0])

        res_fields, vector_down, vector_up, energies, lvl_down, lvl_up = self._precompute_data(res_fields, vector_down,
                                                                                               vector_up, energies,
                                                                                               lvl_down, lvl_up
                                                                                               )
        initial_populations = self._get_initial_populations(energies)
        free, ind, evo, prof = self._matrix_and_profile(energies, lvl_down, lvl_up)
        size = 3000
        time = torch.linspace(-100, 70 * 1e3, size) * 1e-6
        n = self.run(time, initial_populations, prof, evo, free, ind)
        res = self._post_compute_data(initial_populations, n, lvl_down, lvl_up)

        return res
