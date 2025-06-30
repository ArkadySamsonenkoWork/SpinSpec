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
from . import tr_utils


class BaseTimeDependantPopulator:
    def __init__(self,
                 solver: tp.Callable = tr_utils.EvolutionSolver.odeint_solver,
                 start_temp: float = 300,
                 context: dict[str, tp.Any] | None = None):
        """
        :param start_temp: temperature in K
        """
        self.start_temp = torch.tensor(start_temp)
        self.context = context or None
        self.solver = solver

    def _initial_populations(self, energies, lvl_down, lvl_up, *args, **kwargs):
        return nn.functional.softmax(-constants.unit_converter(energies) / self.start_temp, dim=-1)

    def _precompute(self, res_fields, vector_down, vector_up, energies, lvl_down, lvl_up, *args, **kwargs):
        energies = copy.deepcopy(energies)
        return res_fields, vector_down, vector_up, energies, lvl_down, lvl_up

    def _post_compute(self, initial_populations, time_dep_population, lvl_down, lvl_up, *args, **kwargs):
        indexes = torch.arange(initial_populations.shape[-2], device=initial_populations.device)
        time_intensities = time_dep_population[..., indexes, lvl_down] - time_dep_population[..., indexes, lvl_up]
        init_intensity = initial_populations[..., indexes, lvl_down] - initial_populations[..., indexes, lvl_up]
        return time_intensities - init_intensity.unsqueeze(0)

    def _init_tr_matrix_generator(self, res_fields, vector_down,
                                  vector_up, energies,
                                  lvl_down, lvl_up, time, *args, **kwargs) -> tr_utils.TransitionMatrixGenerator:
        tr_matrix_generator = tr_utils.TransitionMatrixGenerator(self.context)
        return tr_matrix_generator

    def __call__(self, res_fields, vector_down, vector_up, energies, lvl_down, lvl_up, time, *args, **kwargs):
        res_fields, vector_down, vector_up, energies, lvl_down, lvl_up = self._precompute(res_fields, vector_down,
                                                                                          vector_up, energies,
                                                                                          lvl_down, lvl_up)

        tr_matrix_generator = self._init_tr_matrix_generator(res_fields, vector_down,
                                                             vector_up, energies,
                                                             lvl_down, lvl_up, time, *args, **kwargs)
        initial_populations = self._initial_populations(energies, lvl_down, lvl_up)
        evo = tr_utils.EvolutionMatrix(energies)
        populations = self.solver(
            time, initial_populations, evo, tr_matrix_generator)
        res = self._post_compute(initial_populations, populations, lvl_down, lvl_up)
        return res