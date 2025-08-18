from abc import ABC, abstractmethod

import torch
from torch import nn

import constants
import copy
import typing as tp

from . import tr_utils


class BaseTimeDependantPopulator(ABC):
    def __init__(self,
                 context: tp.Any,
                 tr_matrix_generator_cls: tp.Type[tr_utils.TransitionMatrixGenerator],
                 solver: tp.Callable = tr_utils.EvolutionVectorSolver.odeint_solver,
                 init_temp: float = 300):
        """
        :param init_temp: temperature in K
        """
        self.init_temp = torch.tensor(init_temp)
        self.context = context or None
        self.solver = solver
        self.tr_matrix_generator_cls = tr_matrix_generator_cls

    def _initial_populations(self, energies, lvl_down, lvl_up, *args, **kwargs):
        return nn.functional.softmax(-constants.unit_converter(energies) / self.init_temp, dim=-1)

    def _precompute(self, res_fields, lvl_down, lvl_up, energies, vector_down, vector_up, *args, **kwargs):
        energies = copy.deepcopy(energies)
        return res_fields, lvl_down, lvl_up, energies, vector_down, vector_up

    def _post_compute(self, initial_populations, time_dep_population, lvl_down, lvl_up, *args, **kwargs):
        indexes = torch.arange(initial_populations.shape[-2], device=initial_populations.device)
        time_intensities = time_dep_population[..., indexes, lvl_down] - time_dep_population[..., indexes, lvl_up]
        init_intensity = initial_populations[..., indexes, lvl_down] - initial_populations[..., indexes, lvl_up]
        return time_intensities - init_intensity.unsqueeze(0)

    @abstractmethod
    def _init_tr_matrix_generator(self, res_fields,
                                  lvl_down, lvl_up, time, energies, vector_down,
                                  vector_up, *args, **kwargs) -> tr_utils.TransitionMatrixGenerator:
        tr_matrix_generator = self.tr_matrix_generator_cls(self.context)
        return tr_matrix_generator

    def __call__(self, time, res_fields, lvl_down, lvl_up, energies, vector_down, vector_up, *args, **kwargs):
        res_fields, lvl_down, lvl_up, energies, vector_down, vector_up = self._precompute(res_fields,
                                                                                          lvl_down, lvl_up,
                                                                                          energies, vector_down,
                                                                                          vector_up)

        tr_matrix_generator = self._init_tr_matrix_generator(res_fields,
                                                             lvl_down, lvl_up, time, energies, vector_down,
                                                             vector_up, *args, **kwargs)

        initial_populations = self._initial_populations(energies, lvl_down, lvl_up)
        evo = tr_utils.EvolutionMatrix(energies)
        populations = self.solver(
            time, initial_populations, evo, tr_matrix_generator)
        res = self._post_compute(initial_populations, populations, lvl_down, lvl_up)
        return res