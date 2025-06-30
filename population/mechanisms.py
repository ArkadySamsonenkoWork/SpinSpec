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
from . import time_population
from .temperature_dependance import profiles


class T1Context(tp.TypedDict):
    zero_vectors: torch.Tensor
    free_probs_transform: torch.Tensor
    free_probs: torch.Tensor

class TimeResolvedContext(tp.TypedDict):
    zero_vectors: torch.Tensor
    free_probs_transform: torch.Tensor
    free_probs: torch.Tensor
    profile: profiles.Profile


class T1TransitionMatrixGenerator(tr_utils.TransitionMatrixGenerator):
    def __init__(self, context: T1Context | None, temp: torch.Tensor, eigen_vectors_full: torch.Tensor):
        super().__init__(context)
        coeffs = transform.get_transformation_coeffs(context["zero_vectors"].unsqueeze(-3), eigen_vectors_full)
        transformed_probs = transform.transform_rates_matrix(context["free_probs_transform"], coeffs)
        probs = transformed_probs + context["free_probs"]
        self.probs = probs
        self.temp = temp

    def _base_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor:
        """Return the free transition probabilities at given temperature(s)"""
        return self.probs

    def _temperature(self, time: torch.Tensor) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return self.temp


class TempDepMatrixGenerator(tr_utils.TransitionMatrixGenerator):
    def __init__(self, context: TimeResolvedContext | None, eigen_vectors_full: torch.Tensor):
        super().__init__(context)
        self.coeffs = transform.get_transformation_coeffs(context["zero_vectors"].unsqueeze(-3), eigen_vectors_full)
        self.free_probs_transform = context["free_probs_transform"]
        self.free_probs = context["free_probs"]
        self.profile = context["profile"]

    def _temperature(self, time: torch.Tensor) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return self.profile(time)[..., None, None, None, None]

    def _base_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor:
        transformed_probs = transform.transform_rates_matrix(
            self.free_probs_transform(temp.squeeze()).unsqueeze(1).unsqueeze(1),
            self.coeffs.unsqueeze(0))
        probs = transformed_probs + self.free_probs(temp.squeeze()).unsqueeze(1).unsqueeze(1)
        return probs


class T1Population(time_population.BaseTimeDependantPopulator):
    def _precompute(self, res_fields, vector_down, vector_up, energies, lvl_down, lvl_up, *args, **kwargs):
        energies = copy.deepcopy(energies)
        return res_fields, vector_down, vector_up, energies, lvl_down, lvl_up

    def _post_compute(self, initial_populations, time_dep_population,
                      lvl_down, lvl_up, *args, **kwargs):
        indexes = torch.arange(initial_populations.shape[-2], device=initial_populations.device)
        time_intensities = time_dep_population[..., indexes, lvl_down] - time_dep_population[..., indexes, lvl_up]
        return time_intensities

    def _initial_populations(self, energies, lvl_down, lvl_up, *args, **kwargs):
        populations = nn.functional.softmax(-constants.unit_converter(energies) / self.start_temp, dim=-1)
        new_populations = copy.deepcopy(populations)
        indexes = torch.arange(energies.shape[-2], device=energies.device)
        new_populations[..., indexes, lvl_down] = populations[..., indexes, lvl_up]
        new_populations[..., indexes, lvl_up] = populations[..., indexes, lvl_down]
        return new_populations

    def _init_tr_matrix_generator(self, res_fields, vector_down,
                                  vector_up, energies, lvl_down, lvl_up, time, *args, **kwargs):
        tr_matrix_generator = T1TransitionMatrixGenerator(self.context, self.start_temp, args[0])
        return tr_matrix_generator


class TempDepTrPopulation(time_population.BaseTimeDependantPopulator):
    def _init_tr_matrix_generator(self, res_fields, vector_down,
                                  vector_up, energies, lvl_down, lvl_up, time, *args, **kwargs):
        tr_matrix_generator = TempDepMatrixGenerator(self.context, args[0])
        return tr_matrix_generator

