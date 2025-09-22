from dataclasses import dataclass
import itertools

import torch
from torch import nn
import constants

import copy
import typing as tp

from . import transform
from . import tr_utils
from . import time_population
from .temperature_dependance import profiles

sample_kinetic = tp.NewType("sample_kinetic", tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor, torch.Tensor,
                                                    torch.Tensor, torch.Tensor,
                                                    torch.Tensor, torch.Tensor, torch.Tensor])


@dataclass
class BaseSampleContext:
    free_probs: tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]


@dataclass
class TripletMechanismContext(BaseSampleContext):
    zero_field_vectors: torch.Tensor
    transform_probs: tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]
    free_probs: tp.Union[torch.Tensor, tp.Callable[[torch.Tensor], torch.Tensor]]


@dataclass
class T1Context(TripletMechanismContext):
    pass


@dataclass
class TempDepContext(TripletMechanismContext):
    profile: profiles.Profile


@dataclass
class KineticContext:
    sample_contexts: list[BaseSampleContext]
    kinetic_rates: list[torch.Tensor]
    concentrations: list[float]


class ConstTempGeneralMechanism(tr_utils.TransitionMatrixGenerator):
    """
    It is constant temperature relaxation mechanism. The most simple Relaxation. It is assumed that for all
    [...] batch dimensions, including orientations the probabilities of transitions are the same.
    """
    def __init__(self, context: BaseSampleContext, temp: torch.Tensor, system_vector: tp.Optional[torch.Tensor] = None,
                 device: torch.device = torch.device("cpu"), *args, **kwargs):
        super().__init__(context, device=device, *args, **kwargs)
        self.free_probs = context.free_probs.to(device)
        self.temp = temp

    def _base_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor:
        """Return the free transition probabilities at given temperature(s)"""
        return self.free_probs

    def _temperature(self, time: torch.Tensor) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return self.temp

class BaseTripletMechanismGenerator(tr_utils.TransitionMatrixGenerator):
    """
    Triplet mechainsm describe the next mechanism. Let's consider the system in the magnetic field and without it.
    Let's denote the eigen vectors in magnetic field as V_new, without field V_old.
    Then In the absence of magnetic field the
    """
    def __init__(self, context: TripletMechanismContext, system_vectors: torch.Tensor,
                 device: torch.device = torch.device("cpu"), *args, **kwargs):
        super().__init__(context, device=device, *args, **kwargs)
        self.system_vectors = system_vectors
        self.context = context
        self.basis_coeffs = transform.get_transformation_coeffs(
            context.zero_field_vectors.to(device).unsqueeze(-3),
            system_vectors
        )

    def _compute_probs(self,
                       free_probs_transform: torch.Tensor,
                       free_probs: torch.Tensor,
                       basis_coeffs: torch.Tensor | None = None) -> torch.Tensor:
        if basis_coeffs is None:
            basis_coeffs = self.basis_coeffs
        transformed_probs = transform.transform_rates_matrix(free_probs_transform, basis_coeffs)
        return transformed_probs + free_probs


class ConstTempTripletMechanismGenerator(BaseTripletMechanismGenerator):
    def __init__(self, context: T1Context | None, temp: torch.Tensor, system_vectors: torch.Tensor,
                 device: torch.device = torch.device("cpu"), *args, **kwargs):
        super().__init__(context, system_vectors, device=device, *args, **kwargs)
        self.temp = temp

        self.probs = self._compute_probs(
            context.transform_probs,
            context.free_probs
        )

    def _base_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor:
        """Return the free transition probabilities at given temperature(s)"""
        return self.probs

    def _temperature(self, time: torch.Tensor) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return self.temp


class TempDepTripletMechanismGenerator(BaseTripletMechanismGenerator):
    def __init__(self, context: TempDepContext, system_vectors: torch.Tensor,
                 device: torch.device = torch.device("cpu"), *args, **kwargs):
        super().__init__(context, system_vectors, device=device, *args, **kwargs)
        self.profile = context.profile
        self.free_probs_transform = context.transform_probs
        self.free_probs = context.free_probs

    def _temperature(self, time: torch.Tensor) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return self.profile(time)[..., None, None, None, None]

    def _base_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor:
        return self._compute_probs(
            self.free_probs_transform(temp.squeeze()).unsqueeze(1).unsqueeze(1),
            self.free_probs(temp.squeeze()).unsqueeze(1).unsqueeze(1),
            self.basis_coeffs.unsqueeze(0)
        )


class TransitionMatrixGeneratorKinetic(tr_utils.BaseMatrixGenerator):
    def __init__(self,
                 context: KineticContext,
                 temp: torch.Tensor,
                 eigen_vectors_list,
                 transition_matrix_generators: list[tr_utils.TransitionMatrixGenerator],
                 device: torch.device = torch.device("cpu"), *args, **kwargs):
        super().__init__(context=context, device=device)
        self.transition_matrix_generators = transition_matrix_generators
        self.num_blocks = len(eigen_vectors_list)
        self.temp = torch.tensor(temp)

        blocks: tp.List[tp.List[torch.Tensor]] = []
        for i, eig_i in enumerate(eigen_vectors_list):
            row_blocks: tp.List[torch.Tensor] = []

            kynetic_rates = context.kinetic_rates[i]
            diag_rate = torch.diag_embed(kynetic_rates)
            diag_rate = self._expand_diag_rates(eig_i, diag_rate)
            row_blocks.append(-diag_rate)
            for j, eig_j in enumerate(eigen_vectors_list):
                if i == j:
                    continue
                coeffs = transform.get_transformation_coeffs(eig_i, eig_j)
                row_blocks.append(coeffs @ diag_rate)
            blocks.append(row_blocks)

        rows: tp.List[torch.Tensor] = []
        for row in blocks:
            rows.append(torch.cat(row, dim=-1))
        self.kynetic_matrix = torch.cat(rows, dim=-2)

    def _expand_diag_rates(self, eig: torch.Tensor, diag_rate: torch.Tensor):
        return diag_rate.expand(eig.shape)

    def _kynetic_transitions(self, time):
        return self.kynetic_matrix

    def _temperature(self, time: torch.Tensor) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return self.temp

    def __call__(self, time: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

        temp = self._temperature(time)

        base_blocks = []
        induced_blocks = []
        outgoing_blocks = []

        induced_has_non_none = False
        outgoing_has_non_none = False

        for gen in self.transition_matrix_generators:
            t, b, i, o = gen(time)
            base_blocks.append(b)

            if i is not None:
                induced_has_non_none = True
            if o is not None:
                outgoing_has_non_none = True

            b_shape = b.shape
            induced_blocks.append(i if i is not None else torch.zeros_like(b))
            outgoing_blocks.append(
                o if o is not None else torch.zeros(
                    (*b_shape[:-2], b_shape[-1]), dtype=b.dtype, device=b.device
                )
            )
        base_blocks = torch.block_diag(*base_blocks)
        induced_blocks = torch.block_diag(*induced_blocks) if induced_has_non_none else None
        outgoing_blocks = torch.cat(outgoing_blocks, dim=-1) if outgoing_has_non_none else None
        return temp, base_blocks, induced_blocks, outgoing_blocks, self._kynetic_transitions(time)


class T1Population(time_population.BaseTimeDependantPopulator):
    """
    Computes the T1 relaxation of a spin system.
    It changes the populations of the transition levels and measure relaxation of population
    """
    def __init__(self,
                 context: T1Context,
                 tr_matrix_generator_cls: tp.Type[tr_utils.TransitionMatrixGenerator] =
                 ConstTempTripletMechanismGenerator,
                 solver: tp.Callable = tr_utils.EvolutionVectorSolver.odeint_solver,
                 init_temp: float = 300, device: torch.device = torch.device("cpu")):
        """
        :param init_temp: temperature in K
        """
        super().__init__(context, tr_matrix_generator_cls, solver, init_temp, device=device)

    def _precompute(self, res_fields, lvl_down, lvl_up, energies, vector_down, vector_up, *args, **kwargs):
        energies = copy.deepcopy(energies)
        return res_fields, lvl_down, lvl_up, energies, vector_down, vector_up

    def _post_compute(self, time_intensities: torch.Tensor, *args, **kwargs):
        """
        :param time_intensities: The population difference between transitioning energy levels depending on time.
            The shape is [time, ....]
        :return: intensity of transitions due to population difference
        """
        return time_intensities

    def _initial_populations(self, energies: torch.Tensor,
                             lvl_down: torch.Tensor,
                             lvl_up: torch.Tensor,
                             *args, **kwargs):
        populations = nn.functional.softmax(-constants.unit_converter(energies, "Hz_to_K") / self.init_temp, dim=-1)
        new_populations = copy.deepcopy(populations)
        indexes = torch.arange(energies.shape[-2], device=energies.device)
        new_populations[..., indexes, lvl_down] = populations[..., indexes, lvl_up]
        new_populations[..., indexes, lvl_up] = populations[..., indexes, lvl_down]
        return new_populations

    def _init_tr_matrix_generator(self,
                                  time: torch.Tensor,
                                  res_fields: torch.Tensor,
                                  lvl_down: torch.Tensor,
                                  lvl_up: torch.Tensor,
                                  vector_down: torch.Tensor,
                                  vector_up: torch.Tensor,
                                  energies: torch.Tensor, *args, **kwargs):
        tr_matrix_generator = self.tr_matrix_generator_cls(self.context, self.init_temp, args[0])
        return tr_matrix_generator


class TempDepTrPopulator(time_population.BaseTimeDependantPopulator):
    """
    Computes population at the case when the parameters of relaxation depend on temperature dn / dt = K(T) @ n
    """
    def __init__(self,
                 context: TempDepContext,
                 tr_matrix_generator_cls: tp.Type[tr_utils.TransitionMatrixGenerator] =
                 TempDepTripletMechanismGenerator,
                 solver: tp.Callable = tr_utils.EvolutionVectorSolver.odeint_solver,
                 init_temp: float = 300, device: torch.device = torch.device("cpu")):
        """
        :param init_temp: temperature in K
        """
        super().__init__(context, tr_matrix_generator_cls, solver, init_temp, device=device)

    def _init_tr_matrix_generator(self, time, res_fields, lvl_down, lvl_up, vector_down,
                                  vector_up, energies, *args, **kwargs):
        tr_matrix_generator = self.tr_matrix_generator_cls(self.context, args[0])
        return tr_matrix_generator

    def _initial_populations(
            self, energies: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            *args, **kwargs
    ):

        return nn.functional.softmax(-constants.unit_converter(energies, "Hz_to_K") / self.init_temp, dim=-1)

    def _post_compute(self, time_intensities: torch.Tensor, *args, **kwargs):
        """
        :param time_intensities: The population difference between transitioning energy levels depending on time.
            The shape is [time, ....]
        :return: intensity of transitions due to population difference
        """
        return time_intensities - time_intensities[0].unsqueeze(0)


class KineticPopulator(time_population.BaseTimeDependantPopulator):
    def __init__(self,
                 context: KineticContext,
                 sample_tr_matrix_generators: list[tp.Type[tr_utils.TransitionMatrixGenerator]],
                 tr_matrix_generator_cls: tp.Type[TransitionMatrixGeneratorKinetic] = TransitionMatrixGeneratorKinetic,
                 solver: tp.Callable = tr_utils.EvolutionVectorSolver.odeint_solver,
                 init_temp: float = 300, device: torch.device = torch.device("cpu")):
        """
        :param init_temp: temperature in K
        """
        super().__init__(context, tr_matrix_generator_cls, solver, init_temp, device=device)
        self.sample_tr_matrix_generators_cls = sample_tr_matrix_generators

    def _compute_eigendata_in_resonance(self, sample_main, sample_additional):
        F_additional, Gz_additional = sample_additional[-2], sample_additional[-1]
        res_fields = sample_main[0]
        H_additional = F_additional.unsqueeze(-3) + \
                       res_fields.unsqueeze(-1).unsqueeze(-1) * Gz_additional.unsqueeze(-3)
        energies, vectors = torch.linalg.eigh(H_additional)
        return energies, vectors

    def _initial_populations(self, energies_seq, *args, **kwargs):
        conc_seq = self.context.concentrations
        populations = [
            conc * nn.functional.softmax(
                -constants.unit_converter(energies, "Hz_to_K") / self.init_temp, dim=-1
            ) for energies, conc in zip(energies_seq, conc_seq)
        ]
        return torch.cat(populations, dim=-1)

    def _init_tr_matrix_generator(self,
                                  eigen_vectors_seq: list[torch.Tensor],
                                  *args, **kwargs
                                  ) -> tr_utils.BaseMatrixGenerator:
        contexts = self.context.sample_contexts
        tr_matrix_generators = []
        for idx, (eigen_vector, context) in enumerate(zip(eigen_vectors_seq, contexts)):
            tr_matrix_generators.append(self.sample_tr_matrix_generators_cls[idx](context,
                                                                                  self.init_temp,
                                                                                  eigen_vector,
                                                                                  eigen_vector.device))
        tr_matrix_generator = self.tr_matrix_generator_cls(self.context,
                                                           self.init_temp,
                                                           eigen_vectors_seq,
                                                           tr_matrix_generators)
        return tr_matrix_generator

    def _compute_relaxation_sample(self, idx: int,
                                   sample_main: sample_kinetic,
                                   samples_reference: list[sample_kinetic],
                                   spin_indexes: list[list[int]],
                                   time: torch.Tensor):
        energies_reference_seq = []
        vectors_reference_seq = []
        for sample_reference in samples_reference:
            energies_additional, vectors_additional = \
                self._compute_eigendata_in_resonance(sample_main, sample_reference)
            energies_reference_seq.append(energies_additional)
            vectors_reference_seq.append(vectors_additional)

        res_fields_main, lvl_down_main, lvl_up_main, energies_main, _, _, vectors_full_main, _, _ = sample_main

        vectors_reference_seq.insert(idx, vectors_full_main)
        energies_reference_seq.insert(idx, energies_main)

        initial_populations = self._initial_populations(energies_reference_seq)

        tr_matrix_generator = self._init_tr_matrix_generator(
            vectors_reference_seq, self.context
        )
        energies = torch.cat(energies_reference_seq, dim=-1)
        evo = tr_utils.EvolutionMatrixKinetic(energies)
        populations = self.solver(
            time, initial_populations, evo, tr_matrix_generator)

        populations = populations[..., spin_indexes[idx]]
        initial_populations = initial_populations[..., spin_indexes[idx]]
        res = self._compute_transition_population_difference(initial_populations, populations,
                                                             lvl_down_main, lvl_up_main)
        res = self._post_compute(res)
        return res

    def _post_compute(self, time_intensities: torch.Tensor, *args, **kwargs):
        """
        :param time_intensities: The population difference between transitioning energy levels depending on time.
            The shape is [time, ....]
        :return: intensity of transitions due to population difference
        """
        return time_intensities - time_intensities[0].unsqueeze(0)

    def _get_spin_indexes(self, spin_dimensions: list[int]) -> list[list[int]]:
        starts = [0] + list(itertools.accumulate(spin_dimensions[:-1]))
        return [list(range(start, start + size)) for start, size in zip(starts, spin_dimensions)]

    def forward(self, time: torch.Tensor,
                 samples: list[sample_kinetic], spin_dimensions: list[int], *args, **kwargs):


        samples_num = len(samples)
        spin_indexes = self._get_spin_indexes(spin_dimensions)
        results = []
        for idx in range(samples_num):
            results.append(
                self._compute_relaxation_sample(
                    idx, samples[idx], samples[:idx] + samples[idx + 1:], spin_indexes, time)
            )
        return results

