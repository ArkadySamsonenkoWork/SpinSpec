import random
from enum import Enum
import typing as tp
import sys
import itertools
import pathlib
import pickle
import tqdm
import traceback
import datetime
from dataclasses import dataclass
sys.path.append("..")

import torch
import safetensors

import spin_system
import particles
import mesher
import constants
from .spectra_generation import GenerationCreator


@dataclass
class SpinSystemStructure:
    num_electrons: int
    num_nuclei: int
    nuclei: list[str]

    electron_nuclei: list[tuple[int, int]]
    electron_electron: list[tuple[int, int]]
    nuclei_nuclei: list[tuple[int, int]]

    nuclei_spins: list[float]
    electrons_spins: list[float]

    bound_map: dict[int, int]
    spin_system_dim: int


class RandomStructureGenerator:
    """
    Generate the structure of a spin system: electrons, nuclei and interaction between them
    Each nucleus has hyperfine interaction with some electron spin
    """
    def __init__(
            self,
            max_num_electrons: int,
            min_num_electrons: int,
            max_num_nuclei: int,
            min_num_nuclei: int,
            max_dim: int,

            max_interactions: tp.Optional[int] = None,
            electron_allowed_spins: tp.Optional[list[float]] = None,
            electron_spin_probabilities: tp.Optional[dict[float, float]] = None,
            nucleus_labels: tp.Optional[list[str]] = None,
            nucleus_labels_probabilities: tp.Optional[dict[float, float]] = None,
            ion_spin_defaults: tp.Optional[dict[str, list[float]]] = None,
            ion_spin_usage_prob: float = 0.0,
            rng_generator: tp.Optional[random.Random] = None,

            electron_nuc_interaction_prob: float = 0.2,
            electron_electron_prob: float = 0.4,
            zfs_probability: float = 0.5,

    ) -> None:
        """
        :param max_num_electrons: maximum number of electrons in structure
        :param min_num_electrons: minimum number of electrons in structure
        :param max_num_nuclei: maximum number of nuclei in structure
        :param min_num_nuclei: minimum number of nuclei in structure
        :param max_dim: maximum dimension of a spin system (total number of energy levels)
        :param max_interactions: Maximum number of interactions.
                - It does not include g-tensor
                - Each nucleus has at least one bond with the electron spin, and the first bond is not counted.
                - None means that all number is possible

        :param electron_allowed_spins: Allowed value of electron spins. Example, [1/2, 1, 3/2].
                Default is None. If it is none then allowed spins: [0.5, 1.0, 1.5, 2.0, 2.5]

        :param electron_spin_probabilities: The probabilities of electrons spins. Example [0.3, 0.4, 0.3].
                Default is None. If it is none then all probabilities are the same

        :param nucleus_labels: The allowed labels of nuclei. Default is None. If it is none then allowed labels:
                ["14N", "1H", "2H", "15N", "63Cu", "65Cu", "59Co", "55Mn", "31P", "19F"]

        :param nucleus_labels_probabilities: The probabilities of electrons spins. Example [0.3, 0.4, 0.3].
                Default is None. If it is none then all probabilities are the same

        :param ion_spin_defaults: In some cases we usually want to specify the spin of the ion,
            rather than the electron for the nucleus.
            For example, if 65Cu was obtained, then we can specify that the electron spin for this nucleus
            should be [1/2, 3/2]
            Example: {'65Cu': [1/2, 3/2]}

        :param ion_spin_usage_prob: If ion_spin_defaults was set, then there is probability to generate the electron
        spin on a given nucleus

        :param rng_generator:

        :param electron_nuc_interaction_prob: Probability of generating a bond between an electron and a nucleus.
            This does not take into account the mandatory random generation of a
            bond between each nucleus and one of the electrons.

        :param electron_electron_prob: Probability of generating a bond between an electron and another electron

        :param zfs_probability: Probability of generating a bond between an electron and the same electron.
                Note! This is possible only the electron spin greater than 1/2
        """
        if rng_generator is None:
            self.rng = random.Random(None)
        else:
            self.rng = rng_generator

        if electron_allowed_spins is None:
            self.electron_allowed_spins = [0.5, 1.0, 1.5, 2.0, 2.5]
        else:
            self.electron_allowed_spins = list(electron_allowed_spins)

        if electron_spin_probabilities is None:
            self.electron_spin_weights = {s: 1.0 for s in self.electron_allowed_spins}
        else:
            self.electron_spin_weights = {float(k): float(v) for k, v in electron_spin_probabilities.items()}
            for s in self.electron_allowed_spins:
                self.electron_spin_weights.setdefault(s, 0.0)
        self.electron_spin_weights = self._normalize_prob_dict(self.electron_spin_weights)

        if nucleus_labels is None:
            self.nucleus_allowed_labels = ["14N", "1H", "2H", "15N", "63Cu", "65Cu", "59Co", "55Mn", "31P", "19F"]
        else:
            self.nucleus_allowed_labels = nucleus_labels

        if nucleus_labels_probabilities is None:
            self.nucleus_labels_weights = {s: 1.0 for s in self.nucleus_allowed_labels}
        else:
            self.nucleus_labels_weights = {float(k): float(v) for k, v in nucleus_labels_probabilities.items()}
            for s in self.nucleus_allowed_labels:
                self.nucleus_labels_weights.setdefault(s, 0.0)
        self.nucleus_labels_weights = self._normalize_prob_dict(self.nucleus_labels_weights)

        self.ion_spin_defaults = ion_spin_defaults or {}
        for k, v in list(self.ion_spin_defaults.items()):
            self.ion_spin_defaults[k] = [float(x) for x in v]

        if not (0.0 <= zfs_probability <= 1.0):
            raise ValueError("zfs_probability must be between 0 and 1")
        if not (0.0 <= ion_spin_usage_prob <= 1.0):
            raise ValueError("ion_spin_usage_prob must be between 0 and 1")

        self.max_num_electrons = max_num_electrons
        self.max_num_nuclei = max_num_nuclei
        self.max_dim = max_dim
        self.max_interactions = max_interactions

        self.zfs_probability = zfs_probability
        self.ion_spin_usage_prob = ion_spin_usage_prob
        self.min_num_electrons = min_num_electrons
        self.min_num_nuclei = min_num_nuclei

        self.electron_nuc_interaction_prob = electron_nuc_interaction_prob
        self.electron_electron_prob = electron_electron_prob

        self.trial = 0

    def _get_nucleus_spin(self, label: str) -> float:
        return float(particles.Nucleus(label).spin)

    def _hilbert_dim_from_spins(self, spins: list[float]) -> int:
        dim = 1
        for s in spins:
            dim *= int(2 * s + 1)
        return dim

    def _normalize_prob_dict(self, prob_dict: dict[float, float]) -> dict[float, float]:
        total = sum(prob_dict.values())
        if total <= 0:
            raise ValueError("Probability weights must sum to a positive value")
        return {k: v / total for k, v in prob_dict.items()}

    def _sample_counts(self) -> tuple[int, int]:
        num_electrons = self.rng.randint(self.min_num_electrons, self.max_num_electrons)
        num_nuclei = self.rng.randint(self.min_num_nuclei, self.max_num_nuclei)
        return num_electrons, num_nuclei

    def _sample_nuclei_labels(self, num_nuclei: int):
        active_nucleus_labels = self.rng.choices(self.nucleus_allowed_labels,
                                                 weights=self.nucleus_labels_weights.values(), k=num_nuclei)
        return active_nucleus_labels

    def _sample_electron_spin_from_weights(self) -> float:
        spins = list(self.electron_spin_weights.keys())
        weights = [self.electron_spin_weights[s] for s in spins]
        return float(self.rng.choices(spins, weights=weights, k=1)[0])

    def _attempt_bind_for_nucleus(self, label: str, electrons_spins: list[float], nuclei_spins: list[float]) -> \
    tp.Optional[float]:
        """Attempt to create a bound electron for `label`. Returns spin if placed, else None."""
        if label not in self.ion_spin_defaults:
            return None
        if self.rng.random() >= self.ion_spin_usage_prob:
            return None
        chosen_spin = float(self.rng.choice(self.ion_spin_defaults[label]))
        if self._hilbert_dim_from_spins(electrons_spins + [chosen_spin] + nuclei_spins) <= self.max_dim:
            return chosen_spin
        for cand in sorted(self.electron_allowed_spins):
            if self._hilbert_dim_from_spins(electrons_spins + [cand] + nuclei_spins) <= self.max_dim:
                return cand
        return None

    def _sample_and_bind_electrons(self, num_electrons: int, nucleus_labels: list[str], nuclei_spins: list[float]) -> \
    tuple[list[float], dict[int, int]]:
        """Create bound electrons for matching nuclei (when possible).


        Returns (electrons_spins, bound_map) where bound_map maps nucleus_idx -> electron_idx.
        Bound electrons count toward num_electrons. The function stops creating bound
        electrons when either all nuclei have been considered or we've reached num_electrons.
        """
        electrons_spins: list[float] = []
        bound_map: dict[int, int] = {}

        for n_idx, label in enumerate(nucleus_labels):
            if len(electrons_spins) >= num_electrons:
                break
            s = self._attempt_bind_for_nucleus(label, electrons_spins, nuclei_spins)
            if s is not None:
                el_idx = len(electrons_spins)
                electrons_spins.append(s)
                bound_map[n_idx] = el_idx
        return electrons_spins, bound_map

    def _fill_remaining_electrons(self, electrons_spins: list[float], nuclei_spins: list[float],
                                  target_num_electrons: int) -> list[float]:
        """Fill the electrons_spins list up to target_num_electrons using weighted sampling, respecting max_dim."""
        while len(electrons_spins) < target_num_electrons:
            s = self._sample_electron_spin_from_weights()
            if self._hilbert_dim_from_spins(electrons_spins + [s] + nuclei_spins) <= self.max_dim:
                electrons_spins.append(s)
            else:
                placed = False
                for cand in sorted(self.electron_allowed_spins):
                    if self._hilbert_dim_from_spins(electrons_spins + [cand] + nuclei_spins) <= self.max_dim:
                        electrons_spins.append(cand)
                        placed = True
                        break
                if not placed:
                    electrons_spins.append(0.5)
        return electrons_spins

    def _assemble_el_nuc_interactions(self, electrons_spins: list[float], nucleus_labels: list[str],
                                      bound_map: dict[int, int], remaining_budget: tp.Optional[int]):
        """
        All nuclei must be connected to some electron spins. This generation does not change the remaining_budget!!!!!
        """
        num_e = len(electrons_spins)
        num_n = len(nucleus_labels)

        electron_nuclei: list[tuple[int, int]] = []
        bounded_nuclei = set()
        bounded_pair = set()
        for n_idx, el_idx in bound_map.items():
            electron_nuclei.append((el_idx, n_idx))
            bounded_nuclei.add(n_idx)
            bounded_pair.add((el_idx, n_idx))

        for n_idx in range(num_n):
            if n_idx in bounded_nuclei:
                continue
            el_idx = self.rng.choice(list(range(num_e)))
            electron_nuclei.append((el_idx, n_idx))
            bounded_nuclei.add(n_idx)
            bounded_pair.add((el_idx, n_idx))

        el_indexes = list(range(num_e))
        self.rng.shuffle(el_indexes)

        n_indexes = list(range(num_n))
        self.rng.shuffle(n_indexes)

        for n_idx in n_indexes:
            for el_idx in el_indexes:
                if (el_idx, n_idx) in bounded_pair:
                    continue
                if self.rng.random() < self.electron_nuc_interaction_prob:
                    electron_nuclei.append((el_idx, n_idx))
                    bounded_pair.add((el_idx, n_idx))
                    if remaining_budget is not None and remaining_budget <= 0:
                        remaining_budget -= 1
            if remaining_budget is not None and  remaining_budget <= 0:
                break
        return electron_nuclei, remaining_budget

    def _assemble_el_el_interactions(self, electrons_spins: list[float], remaining_budget: tp.Optional[int]):
        num_e = len(electrons_spins)
        electron_electron = []
        el_indexes = list(range(num_e))
        self.rng.shuffle(el_indexes)

        for el_idx in el_indexes:
            if electrons_spins[el_idx] == 0.5:
                continue

            if remaining_budget is not None and remaining_budget <= 0:
                break

            if self.rng.random() < self.zfs_probability:
                electron_electron.append((el_idx, el_idx))
                if remaining_budget is not None:
                    remaining_budget -= 1

        bounded_pair = set()
        for start, el_idx_1 in enumerate(el_indexes):
            for el_idx_2 in el_indexes[start:]:
                if el_idx_1 == el_idx_2:
                    continue
                if remaining_budget is not None and remaining_budget <= 0:
                    break
                if (el_idx_1, el_idx_2) in bounded_pair:
                    continue
                if self.rng.random() < self.electron_electron_prob:
                    electron_electron.append((el_idx_1, el_idx_2))
                    bounded_pair.add((el_idx_1, el_idx_2))
                    if remaining_budget is not None:
                        remaining_budget -= 1
        return electron_electron, remaining_budget

    def _assemble_nuc_nuc(self, nucleus_labels: list[str], remaining_budget: tp.Optional[int]):
        num_n = len(nucleus_labels)
        nuclei_nuclei: list[tuple[int, int]] = []
        if remaining_budget is not None and remaining_budget > 0:
            nn_candidates = list(itertools.combinations(range(num_n), 2))
            self.rng.shuffle(nn_candidates)
            take_nn = min(len(nn_candidates), remaining_budget)
            for (i, j) in nn_candidates[:take_nn]:
                if remaining_budget is not None and remaining_budget <= 0:
                    break
                nuclei_nuclei.append((i, j,))
                if remaining_budget is not None:
                    remaining_budget -= 1
        return nuclei_nuclei, remaining_budget

    def _assemble_interactions(self, electrons_spins: list[float], nucleus_labels: list[str],
                               bound_map: dict[int, int]) -> \
            tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], tp.Optional[int]]:
        """Create electron-nucleus (with bound ones first), electron-electron and nucleus-nucleus interactions.
        Returns: (electron_nuclei, electron_electron, nuclei_nuclei, remaining_budget)
        """
        remaining_budget = None if self.max_interactions is None else int(self.max_interactions)
        electron_nuclei, remaining_budget = self._assemble_el_nuc_interactions(electrons_spins, nucleus_labels,
                                                                               bound_map, remaining_budget)
        electron_electron, remaining_budget = self._assemble_el_el_interactions(electrons_spins, remaining_budget)

        nuclei_nuclei, remaining_budget = self._assemble_nuc_nuc(nucleus_labels, remaining_budget)

        return electron_nuclei, electron_electron, nuclei_nuclei, remaining_budget

    def __call__(self) -> SpinSystemStructure:
        self.trial = 0

        num_electrons, num_nuclei = self._sample_counts()
        nucleus_labels = self._sample_nuclei_labels(num_nuclei)
        nuclei_spins = [self._get_nucleus_spin(label) for label in nucleus_labels]

        min_dim = self._hilbert_dim_from_spins([0.5] * (num_electrons + num_nuclei))
        if min_dim > self.max_dim:
            self.trial += 1
            if self.trial > 100:
                raise "Max iteration number has been exceed"
            return self.__call__()
        self.trial = 0

        # attempt bindings
        electrons_spins, bound_map = self._sample_and_bind_electrons(num_electrons, nucleus_labels, nuclei_spins)

        # fill remaining electrons
        electrons_spins = self._fill_remaining_electrons(electrons_spins, nuclei_spins, num_electrons)

        # final dimension check
        final_dim = self._hilbert_dim_from_spins(electrons_spins + nuclei_spins)
        if final_dim > self.max_dim:
            self.trial += 1
            if self.trial > 100:
                raise "Max iteration number has been exceed"
            return self.__call__()
        self.trial = 0

        electron_nuclei, electron_electron, nuclei_nuclei, remaining_budget = self._assemble_interactions(
            electrons_spins, nucleus_labels, bound_map)

        structure = SpinSystemStructure(
            num_electrons=len(electrons_spins),
            num_nuclei=len(nucleus_labels),
            nuclei=nucleus_labels,
            electron_nuclei=electron_nuclei,
            electron_electron=electron_electron,
            nuclei_nuclei=nuclei_nuclei,
            electrons_spins=electrons_spins,
            nuclei_spins=nuclei_spins,
            bound_map=bound_map,
            spin_system_dim=self._hilbert_dim_from_spins(electrons_spins + nuclei_spins),
        )

        return structure

class GenerationMode(Enum):
    SINGLE = "single"
    UNCORRELATED = "uncorrelated"
    ISOTROPIC = "isotropic"
    AXIAL = "axial"
    DE = "DE"

@dataclass
class IsotropicLevel:
    """For isotropic generation: isotropic_value + x,y,z variations"""
    isotropic_bounds: tuple[float, float]
    isotropic_shift: float
    variation_bounds: tuple[float, float]
    variation_shift: float

@dataclass
class AxialLevel:
    """For axial generation: parallel and perpendicular components"""
    parallel_bounds: tuple[float, float]
    parallel_shift: float
    perpendicular_bounds: tuple[float, float]
    perpendicular_shift: float

@dataclass
class UncorrelatedLevel:
    initial_bounds: tp.Union[tuple[float, float]]
    vary_shift: tp.Sequence[float]

@dataclass
class DELevel:
    """For generation of D, E"""
    D_bounds: tp.Union[tuple[float, float]]
    E_attitude_bounds: tp.Union[tuple[float, float]]
    D_shift: float
    E_attitude_shift: float


class MultiDimensionalTensorGenerator:
    """
    Generator of 3 (or 2) dimensioal tensors with repect the bounds of generation.
    Any generation can be splited into 3 levels
    1) Determination of scale of generation. (scale)
    2) Generation of mean parameters among bounds
    3) Generation of tensors itself with some variety close to mean parameter
    """

    def __init__(self,
                 levels: tp.Union[list[UncorrelatedLevel], list[IsotropicLevel], list[AxialLevel], list[DELevel]],
                 output_dims: int = 3,
                 probabilities: tp.Optional[list[float]] = None,
                 rng_generator: tp.Optional[random.Random] = None):
        """
        mode: single, uncorrelated, isotropic, axial, DE.
        """
        if isinstance(levels[0], UncorrelatedLevel):
            self.mode = GenerationMode.UNCORRELATED
        elif isinstance(levels[0], AxialLevel):
            self.mode = GenerationMode.AXIAL
        elif isinstance(levels[0], IsotropicLevel):
            self.mode = GenerationMode.ISOTROPIC
        elif isinstance(levels[0], DELevel):
            self.mode = GenerationMode.DE
        else:
            raise NotImplementedError(f"There is no mode: {type(levels[0])}")

        self.levels = levels
        self.output_dims = output_dims

        if rng_generator is None:
            self.rng = random.Random(None)
        else:
            self.rng = rng_generator

        self.probabilities = self._init_probabilities(probabilities)
        self.uniform_dists = None

        if output_dims not in [1, 3] and (self.mode != GenerationMode.DE):
            raise ValueError("output_dims must be 1 or 3")
        if self.mode == GenerationMode.AXIAL and output_dims == 1:
            raise ValueError("Axial mode requires 3 output dimensions")

    def _init_probabilities(self, probabilities: tp.Optional[list[float]]):
        if probabilities is None:
            probabilities = [1] * len(self.levels)
        sum_prob = sum(probabilities)
        return [prob / sum_prob for prob in probabilities]

    def _generate_uncorrelated(self, active_levels: list[UncorrelatedLevel], device: torch.device):
        """Generate independent x, y, z parameters"""
        dists = []

        for dim in range(self.output_dims):
            lower_bounds = []
            upper_bounds = []

            for level in active_levels:
                min_lim, max_lim = level.initial_bounds[dim]
                mean_param = self.rng.uniform(min_lim, max_lim)

                shift = level.vary_shift[dim] if len(level.vary_shift) > dim else level.vary_shift[0]
                a1 = max(mean_param - shift, level.initial_bounds[dim][0])
                a2 = min(mean_param + shift, level.initial_bounds[dim][1])
                lower_bounds.append(a1)
                upper_bounds.append(a2)

            dist = torch.distributions.Uniform(
                torch.tensor(lower_bounds, device=device),
                torch.tensor(upper_bounds, device=device)
            )
            dists.append(dist)

        return dists

    def _generate_isotropic(self, active_levels: list[IsotropicLevel], device: torch.device):
        """Generate isotropic base + xyz variations"""
        dists = []

        iso_lower_bounds = []
        iso_upper_bounds = []

        for level in active_levels:
            min_lim, max_lim = level.isotropic_bounds
            mean_iso = self.rng.uniform(min_lim, max_lim)

            a1 = max(mean_iso - level.isotropic_shift, level.isotropic_bounds[0])
            a2 = min(mean_iso + level.isotropic_shift, level.isotropic_bounds[1])
            iso_lower_bounds.append(a1)
            iso_upper_bounds.append(a2)

        iso_dist = torch.distributions.Uniform(
            torch.tensor(iso_lower_bounds, device=device),
            torch.tensor(iso_upper_bounds, device=device)
        )

        if self.output_dims == 1:
            return [iso_dist]

        for dim in range(3):
            var_lower_bounds = []
            var_upper_bounds = []

            for level in active_levels:
                min_lim, max_lim = level.variation_bounds
                mean_var = self.rng.uniform(min_lim, max_lim)

                a1 = max(mean_var - level.variation_shift, level.variation_bounds[0])
                a2 = min(mean_var + level.variation_shift, level.variation_bounds[1])
                var_lower_bounds.append(a1)
                var_upper_bounds.append(a2)
            var_dist = torch.distributions.Uniform(
                torch.tensor(var_lower_bounds, device=device),
                torch.tensor(var_upper_bounds, device=device)
            )
            dists.append(var_dist)

        return [iso_dist] + dists

    def _generate_axial(self, active_levels: list[AxialLevel], device: torch.device):
        """Generate parallel and perpendicular components"""
        dists = []

        # Parallel component
        par_lower_bounds = []
        par_upper_bounds = []

        for level in active_levels:
            min_lim, max_lim = level.parallel_bounds
            mean_par = self.rng.uniform(min_lim, max_lim)

            a1 = max(mean_par - level.parallel_shift, level.parallel_bounds[0])
            a2 = min(mean_par + level.parallel_shift, level.parallel_bounds[1])
            par_lower_bounds.append(a1)
            par_upper_bounds.append(a2)

        par_dist = torch.distributions.Uniform(
            torch.tensor(par_lower_bounds, device=device),
            torch.tensor(par_upper_bounds, device=device)
        )
        dists.append(par_dist)

        for _ in range(2):
            perp_lower_bounds = []
            perp_upper_bounds = []

            for level in active_levels:
                min_lim, max_lim = level.perpendicular_bounds
                mean_perp = self.rng.uniform(min_lim, max_lim)

                a1 = max(mean_perp - level.perpendicular_shift, level.perpendicular_bounds[0])
                a2 = min(mean_perp + level.perpendicular_shift, level.perpendicular_bounds[1])
                perp_lower_bounds.append(a1)
                perp_upper_bounds.append(a2)

            perp_dist = torch.distributions.Uniform(
                torch.tensor(perp_lower_bounds, device=device),
                torch.tensor(perp_upper_bounds, device=device)
            )
            dists.append(perp_dist)

        return dists

    def _generate_de(self, active_levels: list[DELevel], device: torch.device):
        """Generate D and E components"""
        dists = []

        # Parallel component
        D_lower_bounds = []
        D_upper_bounds = []

        for level in active_levels:
            min_lim, max_lim = level.D_bounds
            mean_par = self.rng.uniform(min_lim, max_lim)

            a1 = max(mean_par - level.D_shift, level.D_bounds[0])
            a2 = min(mean_par + level.D_shift, level.D_bounds[1])
            D_lower_bounds.append(a1)
            D_upper_bounds.append(a2)

        D_dist = torch.distributions.Uniform(
            torch.tensor(D_lower_bounds, device=device),
            torch.tensor(D_upper_bounds, device=device)
        )
        dists.append(D_dist)

        attitude_lower_bounds = []
        attitude_upper_bounds = []

        for level in active_levels:
            min_lim, max_lim = level.E_attitude_bounds
            mean_perp = self.rng.uniform(min_lim, max_lim)

            a1 = max(mean_perp - level.E_attitude_shift, level.E_attitude_bounds[0])
            a2 = min(mean_perp + level.E_attitude_shift, level.E_attitude_bounds[1])
            attitude_lower_bounds.append(a1)
            attitude_upper_bounds.append(a2)

        E_dist = torch.distributions.Uniform(
            torch.tensor(attitude_lower_bounds, device=device),
            torch.tensor(attitude_upper_bounds, device=device)
        )
        dists.append(E_dist)

        return dists

    def update(self, num_examples: int, device: torch.device = torch.device("cpu")):
        """Update the generator with new random levels"""
        active_levels = self.rng.choices(self.levels, weights=self.probabilities, k=num_examples)

        if self.mode == GenerationMode.UNCORRELATED:
            self.uniform_dists = self._generate_uncorrelated(active_levels, device)
        elif self.mode == GenerationMode.ISOTROPIC:
            self.uniform_dists = self._generate_isotropic(active_levels, device)
        elif self.mode == GenerationMode.AXIAL:
            self.uniform_dists = self._generate_axial(active_levels, device)
        elif self.mode == GenerationMode.DE:
            self.uniform_dists = self._generate_de(active_levels, device)

        return self

    def __call__(self, batch_size: int):
        """Generate samples"""
        if self.uniform_dists is None:
            raise RuntimeError("Must call update() before generating samples")

        if self.mode == GenerationMode.ISOTROPIC and self.output_dims == 3:
            iso_samples = self.uniform_dists[0].sample((batch_size,)).transpose(0, 1)
            var_samples = [dist.sample((batch_size,)).transpose(0, 1) for dist in self.uniform_dists[1:]]
            results = []
            for i, var_sample in enumerate(var_samples):
                results.append(iso_samples + var_sample)

            return torch.stack(results, dim=0)

        elif self.mode == GenerationMode.DE and self.output_dims == 2:
            D_samples = self.uniform_dists[0].sample((batch_size,)).transpose(0, 1)
            E_attitude_samples = self.uniform_dists[1].sample((batch_size,)).transpose(0, 1)
            E_samples = D_samples * E_attitude_samples
            return torch.stack((D_samples, E_samples), dim=0)

        else:
            samples = [dist.sample((batch_size,)).transpose(0, 1) for dist in self.uniform_dists]

            if len(samples) == 1:
                return samples[0]
            else:
                return torch.stack(samples, dim=0)


class MultiOrientedSampleGen(spin_system.MultiOrientedSample):
    def _init_ham_str(self, ham_strain: torch.Tensor):
        if ham_strain is None:
            ham_strain = torch.zeros(
                (*self.base_spin_system.config_shape, 3),
                device=self.device, dtype=torch.float32
            )
        else:
            ham_strain = spin_system.init_tensor(ham_strain, device=self.device, dtype=torch.float32)
        return ham_strain


class SampleGenerator:
    def __init__(self,
                 mesh: mesher.DelaunayMeshNeighbour,
                 structure: SpinSystemStructure,

                 temperature_generator: MultiDimensionalTensorGenerator,
                 hamiltonian_strain_generator: MultiDimensionalTensorGenerator,

                 g_tensor_components_generator: MultiDimensionalTensorGenerator,
                 g_tensor_orientation_generator: MultiDimensionalTensorGenerator,

                 hyperfine_coupling_generator: dict[str, MultiDimensionalTensorGenerator],
                 hyperfine_orientation_generator: MultiDimensionalTensorGenerator,

                 exchange_coupling_generator: MultiDimensionalTensorGenerator,
                 dipolar_coupling_generator: MultiDimensionalTensorGenerator,
                 zero_field_splitting_generator: MultiDimensionalTensorGenerator,
                 electron_electron_orientation_generator: MultiDimensionalTensorGenerator,

                 nuclear_coupling_generator: tp.Optional[MultiDimensionalTensorGenerator] = None,
                 nuclear_orientation_generator: tp.Optional[MultiDimensionalTensorGenerator] = None,

                 num_temperature_points: int = 3,
                 num_hamiltonian_strains: int = 3,
                 ):
        """
        :param mesh: mesh to create crystalline sample
        :param structure: SpinSystemStructure
        :param temperature_generator: Generator to create temperature parameters
        :param hamiltonian_strain_generator: Generator of Hamiltonians Strain
        :param g_tensor_components_generator: Generator of g-tensor-components
        :param g_tensor_orientation_generator: Generator of g-tensor orientations
        :param hyperfine_coupling_generator: Generator of hyperfine coupling interactions
        :param hyperfine_orientation_generator: Generator of hyperfine coupling orientation
        :param exchange_coupling_generator: Generator of exchange coupling. Must return 1 value per batch
        :param dipolar_coupling_generator: Generator of dipolar coupling. Must return 2 value per batch: D and E
        :param zero_field_splitting_generator: Generator of zfs coupling. Must return 2 value per batch: D and E
        :param electron_electron_orientation_generator: Generator of electron-electron interaction orientations

        :param nuclear_coupling_generator:
        :param nuclear_orientation_generator:

        :param num_temperature_points: The number of temperature points to be generated
        :param num_hamiltonian_strains: The number of hamiltonian points to be generated
        Temperature and Hamiltonian Strain have different procedure generation. Per generation, they return not
        batch size elements but num_temperature_points and num_hamiltonian_strains points. To correctly manage with
        hamiltonian_strains the generator uses MultiOrientedSampleGen, not simple MultiOrientedSample
        """

        self.num_electrons = structure.num_electrons
        self.num_nuclei = structure.num_nuclei
        self.nucleus_labels = structure.nuclei
        self.electron_spins = structure.electrons_spins

        self.electron_nucleus_pairs = structure.electron_nuclei
        self.electron_electron_pairs = structure.electron_electron
        self.nucleus_nucleus_pairs = structure.nuclei_nuclei

        self.temperature_gen = temperature_generator
        self.hamiltonian_strain_gen = hamiltonian_strain_generator

        self.g_components_gen = g_tensor_components_generator
        self.g_orientation_gen = g_tensor_orientation_generator

        self.hyperfine_coupling_gen = hyperfine_coupling_generator
        self.hyperfine_orientation_gen = hyperfine_orientation_generator

        self.exchange_coupling_gen = exchange_coupling_generator
        self.dipolar_coupling_gen = dipolar_coupling_generator
        self.zfs_gen = zero_field_splitting_generator
        self.electron_electron_orientation_gen = electron_electron_orientation_generator

        self.nuclear_coupling_gen = nuclear_coupling_generator
        self.nuclear_orientation_gen = nuclear_orientation_generator
        self.num_temp_points = num_temperature_points
        self.num_ham_strains = num_hamiltonian_strains

        self.mesh = mesh

        self.zfs_pairs, self.exchange_dipolar_pairs = self._categorize_electron_electron_pairs()

    def _categorize_electron_electron_pairs(self):
        """Separate electron-electron pairs into ZFS (same electron) and exchange/dipolar (different electrons)"""
        zfs_pairs = []
        exchange_dipolar_pairs = []

        for el_1, el_2 in self.electron_electron_pairs:
            if el_1 == el_2:
                zfs_pairs.append((el_1, el_2))
            else:
                exchange_dipolar_pairs.append((el_1, el_2))

        return zfs_pairs, exchange_dipolar_pairs

    def update(self, device: torch.device = torch.device("cpu")):
        """Update all generators with appropriate batch sizes"""
        self.temperature_gen.update(self.num_temp_points, device)
        self.hamiltonian_strain_gen.update(self.num_ham_strains, device)

        self.g_components_gen.update(self.num_electrons, device)
        self.g_orientation_gen.update(self.num_electrons, device)

        num_hyperfine_pairs = len(self.electron_nucleus_pairs)
        for nucleus_type, generator in self.hyperfine_coupling_gen.items():
            generator.update(num_hyperfine_pairs, device)
        self.hyperfine_orientation_gen.update(num_hyperfine_pairs, device)

        num_exchange_dipolar = len(self.exchange_dipolar_pairs)
        num_zfs = len(self.zfs_pairs)

        if num_exchange_dipolar > 0:
            self.exchange_coupling_gen.update(num_exchange_dipolar, device)
            self.dipolar_coupling_gen.update(num_exchange_dipolar, device)

        if num_zfs > 0:
            self.zfs_gen.update(num_zfs, device)
        total_ee_pairs = len(self.electron_electron_pairs)
        if total_ee_pairs > 0:
            self.electron_electron_orientation_gen.update(total_ee_pairs, device)
        if self.nuclear_coupling_gen is not None:
            num_nuclear_pairs = len(self.nucleus_nucleus_pairs)
            self.nuclear_coupling_gen.update(num_nuclear_pairs, device)

        if self.nuclear_orientation_gen is not None:
            num_nuclear_pairs = len(self.nucleus_nucleus_pairs)
            self.nuclear_orientation_gen.update(num_nuclear_pairs, device)

    def _create_tensor_interactions(self,
                                    components_gen: MultiDimensionalTensorGenerator,
                                    orientation_gen: MultiDimensionalTensorGenerator,
                                    batch_size: int,
                                    device: torch.device) -> \
            tuple[list[spin_system.Interaction], torch.Tensor, torch.Tensor]:
        """Generic method to create tensor interactions with components and orientations
        Returns: (interactions, components_tensor, orientations_tensor)
        """
        components = components_gen(batch_size).transpose(-3, -1).to(device)
        orientations = orientation_gen(batch_size).transpose(-3, -1).to(device)

        interactions = []
        num_entities = components.shape[1]

        for i in range(num_entities):
            interaction = spin_system.Interaction(
                components=components[:, i, :],  # [batch_size, 3]
                frame=orientations[:, i, :],  # [batch_size, 3]
                device=device,
                dtype=components.dtype
            )
            interactions.append(interaction)
        return interactions, components, orientations

    def _assemble_g_tensors(self, batch_size: int, device: torch.device) ->\
            tuple[list[spin_system.Interaction], torch.Tensor, torch.Tensor]:
        """Assemble G-tensor interactions for each electron
        Returns: (interactions, g_components, g_orientations)
        """
        return self._create_tensor_interactions(
            self.g_components_gen,
            self.g_orientation_gen,
            batch_size,
            device
        )

    def _assemble_hyperfine_interactions(self, batch_size: int, device: torch.device) ->\
            tuple[list[tuple[int, int, spin_system.Interaction]], torch.Tensor, torch.Tensor]:
        """Assemble hyperfine interactions for electron-nucleus pairs
        Returns: (interactions, hyperfine_components, hyperfine_orientations)
        """
        first_nucleus_type = list(self.hyperfine_coupling_gen.keys())[0]
        coupling_gen = self.hyperfine_coupling_gen[first_nucleus_type]

        interactions, components, orientations = self._create_tensor_interactions(
            coupling_gen,
            self.hyperfine_orientation_gen,
            batch_size,
            device
        )
        interaction_list = [(el_idx, nuc_idx, interaction) for (el_idx, nuc_idx), interaction in
                            zip(self.electron_nucleus_pairs, interactions)]
        return interaction_list, components, orientations

    def _convert_DE_to_tensor_components(self,
                                         D: torch.Tensor,
                                         E: torch.Tensor) -> torch.Tensor:
        """Convert D and E ratio to tensor components [Dxx, Dyy, Dzz]"""
        Dxx = -D / 3 + E
        Dyy = -D / 3 - E
        Dzz = 2 * D / 3

        return torch.stack([Dxx, Dyy, Dzz], dim=-1)

    def _assemble_electron_electron_interactions(self, batch_size: int, device: torch.device) -> tuple[list, dict]:
        """Assemble all electron-electron interactions (ZFS + exchange + dipolar)
        Returns: (interactions, meta_dict)
        """
        interactions = []
        meta = {}

        if len(self.electron_electron_pairs) > 0:
            orientations = self.electron_electron_orientation_gen(batch_size).transpose(-3, -1).to(device)
            meta['electron_electron_orientations'] = orientations  # [batch_size, num_pairs, 3]
        else:
            meta['electron_electron_orientations'] = torch.empty(batch_size, 0, 3, device=device)

        pair_idx = 0

        if len(self.zfs_pairs) > 0:
            zfs_DE = self.zfs_gen(batch_size)
            D_zfs = zfs_DE[0].to(device).transpose(0, 1)
            E_zfs = zfs_DE[1].to(device).transpose(0, 1)  # [batch_size, num_zfs_pairs]
            zfs_components_list = []

            for i, (el1, el2) in enumerate(self.zfs_pairs):

                zfs_components = self._convert_DE_to_tensor_components(
                    D_zfs[:, i], E_zfs[:, i]
                )  # [batch_size, 3]
                zfs_components_list.append(zfs_components)

                interaction = spin_system.Interaction(
                    components=zfs_components,
                    frame=orientations[:, pair_idx, :],
                    device=device,
                    dtype=zfs_components.dtype
                )
                interactions.append((el1, el2, interaction))
                pair_idx += 1

            if zfs_components_list:
                meta['zfs_components'] = torch.stack(zfs_components_list, dim=1).contiguous()
        else:
            meta['zfs_components'] = torch.empty(batch_size, 0, 3, device=device).contiguous()

        dipolar_components_list = []
        if len(self.exchange_dipolar_pairs) > 0:
            J_values = self.exchange_coupling_gen(batch_size).to(device)

            dipolar_DE = self.dipolar_coupling_gen(batch_size)
            D_dipolar = dipolar_DE[0].to(device).transpose(0, 1)
            E_dipolar = dipolar_DE[1].to(device).transpose(0, 1)

            for i, (el1, el2) in enumerate(self.exchange_dipolar_pairs):
                dipolar_components = self._convert_DE_to_tensor_components(
                    D_dipolar[:, i], E_dipolar[:, i]
                )
                dipolar_components_list = []
                dipolar_components += J_values[:, i].unsqueeze(-1)
                dipolar_components_list.append(dipolar_components)

                interaction = spin_system.Interaction(
                    components=dipolar_components,
                    frame=orientations[:, pair_idx, :],
                    device=device,
                    dtype=dipolar_components.dtype
                )
                interactions.append((el1, el2, interaction))
                pair_idx += 1

            if dipolar_components_list:
                meta['dipolar_components'] = torch.stack(dipolar_components_list, dim=1).contiguous()
        else:
            meta['dipolar_components'] = torch.empty(batch_size, 0, 3, device=device).contiguous()

        return interactions, meta

    def _assemble_spin_system(self, batch_size: int, device: torch.device) -> tuple[spin_system.SpinSystem, dict]:
        """Assemble the complete spin system and return meta information
        Returns: (spin_system, meta_dict)
        """
        meta = {}

        g_tensors, g_components, g_orientations = self._assemble_g_tensors(batch_size, device)
        meta['g_tensor_components'] = g_components.contiguous()
        meta['g_tensor_orientations'] = g_orientations.contiguous()

        hyperfine_interactions, hf_components, hf_orientations = self._assemble_hyperfine_interactions(batch_size,
                                                                                                       device)
        meta['hyperfine_coupling_components'] = hf_components.contiguous()
        meta['hyperfine_coupling_orientations'] = hf_orientations.contiguous()

        electron_electron_interactions, ee_meta = self._assemble_electron_electron_interactions(batch_size, device)
        meta.update(ee_meta)

        if self.nuclear_coupling_gen is not None and len(self.nucleus_nucleus_pairs) > 0:
            nuclear_components = self.nuclear_coupling_gen(batch_size).transpose(-3, -1).to(device)
            meta['nuclear_coupling_components'] = nuclear_components.contiguous()
        else:
            meta['nuclear_coupling_components'] = torch.empty(batch_size, 0, 3, device=device).contiguous()

        if self.nuclear_orientation_gen is not None and len(self.nucleus_nucleus_pairs) > 0:
            nuclear_orientations = self.nuclear_orientation_gen(batch_size).transpose(-3, -1).to(device)
            meta['nuclear_coupling_orientations'] = nuclear_orientations.contiguous()
        else:
            meta['nuclear_coupling_orientations'] = torch.empty(batch_size, 0, 3, device=device).contiguous()
        spin_sys = spin_system.SpinSystem(
            electrons=self.electron_spins,
            g_tensors=g_tensors,
            nuclei=self.nucleus_labels,
            electron_nuclei=hyperfine_interactions,
            electron_electron=electron_electron_interactions,
            device=device
        )

        return spin_sys, meta

    def __call__(self, batch_size: int, device: torch.device = torch.device("cpu")) -> tuple[
        spin_system.MultiOrientedSample, torch.Tensor, dict[str, tp.Any]]:
        """Generate a batch of spin systems with parameters and meta information
        Returns: (multi_oriented_sample, temperatures, meta_dict)
        """
        system_inf = {}
        base_spin_system, system_data = self._assemble_spin_system(batch_size, device)
        system_inf["data"] = system_data

        hamiltonian_strain = self.hamiltonian_strain_gen(1)[:, :, 0]
        hamiltonian_strain = hamiltonian_strain.transpose(-2, -1).to(device)
        temperatures = self.temperature_gen(1)[:, 0].to(device)

        system_inf["data"]['hamiltonian_strain'] = hamiltonian_strain[:, None, None, :].contiguous()
        system_inf["data"]['temperatures'] = temperatures[None, :, None, None].contiguous()

        system_inf["meta"] = {}
        system_inf["meta"]['electron_nucleus_pairs'] = self.electron_nucleus_pairs
        system_inf["meta"]['electron_electron_pairs'] = self.electron_electron_pairs
        system_inf["meta"]['nucleus_nucleus_pairs'] = self.nucleus_nucleus_pairs
        system_inf["meta"]['zfs_pairs'] = self.zfs_pairs
        system_inf["meta"]['exchange_dipolar_pairs'] = self.exchange_dipolar_pairs
        system_inf["meta"]['electron_spins'] = self.electron_spins
        system_inf["meta"]['nucleus_labels'] = self.nucleus_labels

        multi_oriented_sample = MultiOrientedSampleGen(
            spin_system=base_spin_system,
            ham_strain=hamiltonian_strain,
            mesh=self.mesh,
            device=device
        )

        return multi_oriented_sample, temperatures, system_inf


class DataFullGenerator:
    def __init__(self,
                 path: str,
                 struct_generator: RandomStructureGenerator,
                 mesh: mesher.DelaunayMeshNeighbour,

                 freq_generator: MultiDimensionalTensorGenerator,
                 temperature_generator: MultiDimensionalTensorGenerator,
                 hamiltonian_strain_generator: MultiDimensionalTensorGenerator,

                 g_tensor_components_generator: MultiDimensionalTensorGenerator,
                 g_tensor_orientation_generator: MultiDimensionalTensorGenerator,

                 hyperfine_coupling_generator: dict[str, MultiDimensionalTensorGenerator],
                 hyperfine_orientation_generator: MultiDimensionalTensorGenerator,

                 exchange_coupling_generator: MultiDimensionalTensorGenerator,
                 dipolar_coupling_generator: MultiDimensionalTensorGenerator,
                 zero_field_splitting_generator: MultiDimensionalTensorGenerator,
                 electron_electron_orientation_generator: MultiDimensionalTensorGenerator,

                 nuclear_coupling_generator: tp.Optional[MultiDimensionalTensorGenerator] = None,
                 nuclear_orientation_generator: tp.Optional[MultiDimensionalTensorGenerator] = None,

                 num_temperature_points: int = 4,
                 num_hamiltonian_strains: int = 3,
                 fields_base_range: tuple[float, float] = (
                 (constants.PLANCK / (1.9 * constants.BOHR)) / 4, (constants.PLANCK / (2.4 * constants.BOHR)) * 4)
                 ):
        self.base_path = pathlib.Path(path)
        self.struct_generator = struct_generator
        self.temperature_gen = temperature_generator
        self.hamiltonian_strain_gen = hamiltonian_strain_generator

        self.g_components_gen = g_tensor_components_generator
        self.g_orientation_gen = g_tensor_orientation_generator

        self.hyperfine_coupling_gen = hyperfine_coupling_generator
        self.hyperfine_orientation_gen = hyperfine_orientation_generator

        self.exchange_coupling_gen = exchange_coupling_generator
        self.dipolar_coupling_gen = dipolar_coupling_generator
        self.zfs_gen = zero_field_splitting_generator
        self.electron_electron_orientation_gen = electron_electron_orientation_generator

        self.nuclear_coupling_gen = nuclear_coupling_generator
        self.nuclear_orientation_gen = nuclear_orientation_generator
        self.num_temp_points = num_temperature_points
        self.num_ham_strains = num_hamiltonian_strains

        self.mesh = mesh
        self.freq_generator = freq_generator
        self.fields_base_range = torch.tensor([fields_base_range[0], fields_base_range[1]])

    def _ensure_dir(self, p: tp.Union[str, pathlib.Path]):
        p = pathlib.Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _save_pickle(self, obj, target_path: pathlib.Path, filename: str):
        with open(target_path / filename, "wb") as f:
            pickle.dump(obj, f)

    def _save_structure_files(self, struct_obj, target_path: pathlib.Path):
        """
        Try to save easily serializable representation (json) and a pickle backup for the structure.
        """
        self._save_pickle(struct_obj, target_path, "structure.pkl")

    def _first_level_generation(self, structure_index: int, batch_size: int) -> tuple[
        SampleGenerator, pathlib.Path, SpinSystemStructure]:
        """
        Generates a structure via struct_generator(), creates a folder for it and returns
        a configured SampleGenerator for that structure.
        """
        struct = self.struct_generator()
        struct_folder = self.base_path / f"structure_{structure_index:04d}"
        self._ensure_dir(struct_folder)

        self._save_structure_files(struct, struct_folder)

        sample_gen = SampleGenerator(
            mesh=self.mesh,
            structure=struct,
            temperature_generator=self.temperature_gen,
            hamiltonian_strain_generator=self.hamiltonian_strain_gen,
            g_tensor_components_generator=self.g_components_gen,
            g_tensor_orientation_generator=self.g_orientation_gen,
            hyperfine_coupling_generator=self.hyperfine_coupling_gen,
            hyperfine_orientation_generator=self.hyperfine_orientation_gen,
            exchange_coupling_generator=self.exchange_coupling_gen,
            dipolar_coupling_generator=self.dipolar_coupling_gen,
            zero_field_splitting_generator=self.zfs_gen,
            electron_electron_orientation_generator=self.electron_electron_orientation_gen,
            nuclear_coupling_generator=self.nuclear_coupling_gen,
            nuclear_orientation_generator=self.nuclear_orientation_gen,
            num_temperature_points=self.num_temp_points,
            num_hamiltonian_strains=self.num_ham_strains,
        )

        gen_summary = {
            "num_temperature_points": self.num_temp_points,
            "num_hamiltonian_strains": self.num_ham_strains,
            "batch_size": batch_size,
            "intital_mesh_size": self.mesh.initial_grid_frequency,
            "interpolate_mesh_size": self.mesh.interpolation_grid_frequency,
        }
        with open(struct_folder / "generator_summary.pkl", "wb") as f:
            pickle.dump(gen_summary, f)

        return sample_gen, struct_folder, struct

    def _get_freq_field(self, batch_size: int):
        freq = self.freq_generator(1)[0, 0]
        fields = (self.fields_base_range * freq).expand(batch_size, 2)
        return freq, fields

    @torch.inference_mode()
    def generate(self,
                 struct_iterations: int,
                 mean_iterations: int,
                 vary_iterations: int,
                 batch_size: int,
                 device: torch.device = torch.device("cpu")) -> None:
        """
        Run the three-level generation and save outputs.

        - struct_iterations: how many different structures (top level)
        - mean_iterations: how many times to update (mid level) per structure
        - vary_iterations: how many distinct generated samples (low level) per mean iteration

        Files layout:
        base_path/
            structure_0000/
                structure.pkl
                structure_summary.pkl
                generator_summary.pkl
                mean_0000/
                    sample_0000/
                        generation_data.safetensors
                        sample_meta.pkl
                    ...
                mean_0001/
                    ...
        """
        self.fields_base_range.to(device=device)
        for s_idx in range(struct_iterations):
            print(f"structure_iteration {s_idx} / {struct_iterations}")
            sample_gen, struct_folder, struct_obj = self._first_level_generation(s_idx, batch_size)
            for m_idx in range(mean_iterations):
                print(f"mean_iteration {m_idx} / {mean_iterations}")
                mean_folder = struct_folder / f"mean_{m_idx:04d}"
                self._ensure_dir(mean_folder)

                # Update generators for this mean iteration
                sample_gen.update(device=device)
                self.freq_generator.update(1, device=device)

                for v_idx in tqdm.tqdm(range(vary_iterations)):
                    sample_folder = mean_folder / f"sample_{v_idx:04d}"
                    self._ensure_dir(sample_folder)

                    try:
                        multi_oriented_sample, temperatures, system_meta = sample_gen(batch_size=batch_size,
                                                                                      device=device)

                        freq, fields = self._get_freq_field(batch_size=batch_size)
                        creator = GenerationCreator(freq=freq, sample=multi_oriented_sample,
                                                    temperature=temperatures, device=device)
                        out, (min_pos_batch, max_pos_batch) = creator(fields=fields, sample=multi_oriented_sample)

                        save = system_meta["data"]
                        save["fields"] = fields
                        save["out"] = out
                        save["freq"] = freq
                        save["min_field_pos"] = min_pos_batch
                        save["max_field_pos"] = max_pos_batch
                        safetensors.torch.save_file(save, sample_folder / "generation_data.safetensors")

                        with open(sample_folder / "sample_meta.pkl", "wb") as f:
                            pickle.dump(system_meta["meta"], f)

                    except Exception as error:
                        # Log error and break to continue with new structure
                        error_msg = f"Error at structure {s_idx}, mean {m_idx}, vary {v_idx}: {str(error)}\n"
                        error_msg += f"Traceback: {traceback.format_exc()}\n"
                        error_msg += f"Timestamp: {datetime.datetime.now()}\n\n"
                        error_file = mean_folder / f"generation_errors_{s_idx}_{m_idx}_{v_idx}.log"
                        with open(error_file, "a") as f:
                            f.write(error_msg)
                        print(f"Error logged, continuing with new mean values...")
                        break
                else:
                    continue
            else:
                continue