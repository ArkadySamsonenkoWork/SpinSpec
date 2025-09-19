from abc import ABC, abstractmethod

import torch
from torch import nn

import constants
import copy
import typing as tp

from . import tr_utils


class BaseTimeDependantPopulator(nn.Module, ABC):
    """
    Base Time Dependent Populator. To compute the relaxation the following entities should be defined:
    1) Populator itself. Populator determines the population of initial states and how intensity is determined.
    2) Context is all data that are needed to compute relaxation
    3) Matrix_generator: It is an object that returns the probabilities of transitions depending on time:
    """

    def __init__(self,
                 context: tp.Any,
                 tr_matrix_generator_cls: tp.Type[tr_utils.BaseMatrixGenerator],
                 solver: tp.Callable = tr_utils.EvolutionVectorSolver.odeint_solver,
                 init_temp: float = 300,
                 device: torch.device = torch.device("cpu")):
        """
        :param context: context is a dataclass / Dict with any objects that are used to compute relaxation matrix.
        :param tr_matrix_generator_cls: class of Matrix Generator
            that will be used to compute probabilities of transitions
        :param solver: It solves the general equation dn/dt = A(n,t) @ n.

            The following solvers are available:
            - odeint_solver:  Default solver.
            It uses automatic control of time-steps. If you are not sure about the correct time-steps use it
            - stationary_rate_solver. When A does not depend on time use it.
            It just uses that in this case n(t) = exp(At) @ n0
            - exponential_solver. When A does depend on time but does not depend on n,
            It is possible to precompute A and exp(A) in all points.
            In this case the solution is n_i+1 = exp(A_idt) @ ni
        :param init_temp: initial temperature. In default case it is used to find initial population

        :param device: device to compute (cpu / gpu)
        """
        super().__init__()
        self.init_temp = torch.tensor(init_temp)
        self.context = context or None
        self.solver = solver
        self.tr_matrix_generator_cls = tr_matrix_generator_cls
        self.to(device)

    @abstractmethod
    def _initial_populations(
            self, energies: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            *args, **kwargs
    ):
        """
        :param energies:
            The energies of spin states. The shape is [..., N]

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param args:
        :param kwargs:
        :return: initial populations at t=0.
        """
        return nn.functional.softmax(-constants.unit_converter(energies, "Hz_to_K") / self.init_temp, dim=-1)

    def _precompute(self, res_fields, lvl_down, lvl_up, energies, vector_down, vector_up, *args, **kwargs):
        energies = copy.deepcopy(energies)
        return res_fields, lvl_down, lvl_up, energies, vector_down, vector_up

    def _compute_transition_population_difference(
            self, initial_populations: torch.Tensor,
            time_dep_population: torch.Tensor,
            lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            *args, **kwargs
        ):
        """
        Calculate the population difference between transitioning energy levels.

        Parameters
        ----------
        :param initial_populations:
            Population at the initial moment of time.
            Shape: [..., N], where N is the number of energy levels.

        :param time_dep_population:
            Time-dependent population values.
            Shape: [..., N], where N is the number of energy levels.

        :param lvl_down : array-like
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up : array-like
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param args:

        :param kwargs:

        :return:
        -------
            The population difference between transitioning energy levels.
        """
        indexes = torch.arange(initial_populations.shape[-2], device=initial_populations.device)
        time_intensities = time_dep_population[..., indexes, lvl_down] - time_dep_population[..., indexes, lvl_up]
        return time_intensities

    @abstractmethod
    def _post_compute(self, time_intensities: torch.Tensor, *args, **kwargs):
        """
        :param time_intensities: The population difference between transitioning energy levels depending on time
        :return: intensity of transitions due to population difference
        """
        return time_intensities - time_intensities[0].unsqueeze(0)

    @abstractmethod
    def _init_tr_matrix_generator(self,
                                  time: torch.Tensor,
                                  res_fields: torch.Tensor,
                                  lvl_down: torch.Tensor,
                                  lvl_up: torch.Tensor, energies: torch.Tensor,
                                  vector_down: torch.Tensor,
                                  vector_up: torch.Tensor, *args, **kwargs) -> tr_utils.BaseMatrixGenerator:
        """
        Function creates TransitionMatrixGenerator - it is object that can compute probabilities of transitions.
        ----------
        :param time:
            Time points of measurements.

        :param res_fields:
            Resonance fields of transitions.
            Shape: [..., M], where M is the number of resonance energies.

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param energies:
            The energies of spin states. The shape is [..., N]

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param args: tuple, optional.
        If the resfield algorithm returns full_system_vectors the full_system_vectors = args[0]

        :param kwargs : dict, optional

        :param return:
        -------
        TransitionMatrixGenerator instance
        """
        tr_matrix_generator = self.tr_matrix_generator_cls(self.context)
        return tr_matrix_generator

    def forward(self, time: torch.Tensor, res_fields: torch.Tensor,
                 lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                 energies: torch.Tensor, vector_down: torch.Tensor,
                 vector_up: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        :param time:
            Time points of measurements. The shpase is [T], where T is number of time-steps

        :param res_fields:
            Resonance fields of transitions.
            Shape: [..., M], where M is the number of resonance energies.

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param energies:
            The energies of spin states. The shape is [..., N]

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param args: args from spectra creator.
        For example,if CoupledSpectraCreator is used  args[0] is full_system eigen vectors. The shape is [...., M, N, N]
        For some mechanisms not only eigen vectors of resonance levels are needed but eigen vectors of all states.

        :param kwargs:
        :return: Part of the transition intensity that depends on the population of the levels.
        The shape is [T, ...., Tr]
        """
        res_fields, lvl_down, lvl_up, energies, vector_down, vector_up = self._precompute(res_fields,
                                                                                          lvl_down, lvl_up,
                                                                                          energies, vector_down,
                                                                                          vector_up)

        tr_matrix_generator = self._init_tr_matrix_generator(time, res_fields,
                                                             lvl_down, lvl_up, energies, vector_down,
                                                             vector_up, *args, **kwargs)

        initial_populations = self._initial_populations(energies, lvl_down, lvl_up)
        evo = tr_utils.EvolutionMatrix(energies)
        populations = self.solver(
            time, initial_populations, evo, tr_matrix_generator)
        res = self._compute_transition_population_difference(initial_populations, populations, lvl_down, lvl_up)
        res = self._post_compute(res)
        return res