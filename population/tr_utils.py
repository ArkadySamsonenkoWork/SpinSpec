from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torchdiffeq import odeint


import constants
import typing as tp


class BaseMatrixGenerator(ABC):
    @abstractmethod
    def __init__(self, context: tp.Any, device: torch.device = torch.device("cpu"), *args, **kwargs):
        """
        :param context: Metadata describing the relaxation process. It is recommended to use dataclass
        :param args:
        :param kwargs:
        """
        super().__init__()
        self.context = context

    @abstractmethod
    def __call__(self, time: torch.Tensor):
        pass


class TransitionMatrixGenerator(BaseMatrixGenerator):
    """
    Abstract base class for generating transition probability matrices in a two-level (or multi-level)
    system with populations and energy differences.
    The system of rate equations for two levels with populations n1, n2 and energies E1, E2 is:
        dn1/dt = -out_1 - k1 * n1 + k2 * n2
        dn2/dt = -out_2 + k1 * n1 - k2 * n2

    which can be written in matrix form:

        dN/dt = -OUT + K @ N
    where:
      - OUT is a vector of outgoing transitions from the system,
      - K is the relaxation matrix:
            K = [[-k1,  k2],
                 [ k1, -k2]]

    K itself can be rewritten wia K' and induced transition Ind
    K = K' + Ind, where
        K'   – equilibrium relaxation (thermal),
        Ind  – induced transitions,

    At thermal equilibrium, transition rates satisfy detailed balance:
        k'1 / k'2 = n'2 / n'1 = exp(-(E2 - E1) / kT)
    Defining the average relaxation rate:

        k' = (k'1 + k'2) / 2

    we can compute:
        k'2 = 2k' / (1 + exp(-(E2 - E1) / kT))
        k'1 = 2k' * exp(-(E2 - E1) / kT) / (1 + exp(-(E2 - E1) / kT))

    In symmetric form, the "free probabilities" matrix (i.e. mean equilibrium transition probabilities) is:

        base_probs= [[0,  k'],
                    [k', 0]]

    Ind matrix is always symmetry: [[0,  i],
                                    [i, 0]]

    Note!
    Computation of energy-temperature factor should occur in evolution matrix.
    """
    def __init__(self, context: tp.Any, device: torch.device = torch.device("cpu"), *args, **kwargs):
        """
        :param context: Metadata describing the relaxation process. It is recommended to use dataclass
        :param args:
        :param kwargs:
        """
        super().__init__(context=context, device=device)
        self.context = context

    def __call__(self, time: torch.Tensor) ->\
            tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Evaluate transition probabilities at given measurement times.
        Parameters
        :param time: torch.Tensor
        :return: tuple
            (temperature, base_probs, induced_probs, outgoing_probs)
            - temperature : torch.Tensor or None
                System temperature(s) at the given time(s).
            - free_probs : torch.Tensor [..., N, N]
                Thermal equilibrium (Boltzmann-weighted) transition probabilities.

            Example in symmetry form:
                free_probs = [[0,  k'],
                            [k', 0]]

            - induced_probs : torch.Tensor [..., N, N] or None
                Probabilities of induced transitions (e.g. due to external driving).

                Ind matrix is always symmetry: [[0,  i],
                                                [i, 0]]

            - outgoing_probs : torch.Tensor [..., N]  or None
                Out-of-system transition probabilities (loss terms).

        """
        temp = self._temperature(time)
        base_probs = self._base_transition_probs(temp)
        induced = self._induced_transition_probs(temp)
        outgoing = self._outgoing_transition_probs(temp)
        return temp, base_probs, induced, outgoing

    def _temperature(self, time: torch.Tensor) -> torch.Tensor | None:
        """Return temperature(s) at times t"""
        return None

    @abstractmethod
    def _base_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor:
        """"""
        pass

    def _induced_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor | None:
        """Optional induced transitions; default None"""
        return None

    def _outgoing_transition_probs(self, temp: torch.Tensor | None) -> torch.Tensor | None:
        """Optional outgoing transitions; default None"""
        return None


class EvolutionMatrix:
    """
    Construct full evolution matrix from energy differences and transition probabilities.
    """
    def __init__(self, res_energies: torch.Tensor, symmetry_probs: bool = True):
        """
        :param res_energies: The resonance energies. The shape is [..., N, N], where N is spin system dimension
        :param symmetry_probs: Is the probabilities of transitions are given in symmetric form. Default is True
        """
        self.energy_diff = res_energies.unsqueeze(-2) - res_energies.unsqueeze(-1)
        self.energy_diff = constants.unit_converter(self.energy_diff, "Hz_to_K")
        self.config_dim = self.energy_diff.shape[:-2]
        self._probs_matrix = self._prob_matrix_factory(symmetry_probs)

    def _prob_matrix_factory(self, symmetry_probs: bool) -> tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if symmetry_probs:
            return self._compute_boltzmann_symmetry
        else:
            return self._compute_boltzmann_complement

    def _compute_energy_factor(self, temp: torch.Tensor) -> torch.Tensor:
        denom = 1 + torch.exp(-self.energy_diff / temp)  # Must be speed up via batching of temperature
        return torch.reciprocal(denom)

    def _compute_boltzmann_symmetry(self, temp: torch.tensor, free_probs: torch.Tensor) -> torch.Tensor:
        energy_factor = self._compute_energy_factor(temp)
        probs_matrix = 2 * free_probs * energy_factor
        return probs_matrix

    def _compute_boltzmann_complement(self, temp: torch.tensor, free_probs: torch.Tensor) -> torch.Tensor:
        numerator = torch.exp(self.energy_diff / temp)
        probs_matrix = torch.where(free_probs == 0, free_probs.transpose(-1, -2) * numerator, free_probs)
        return probs_matrix

    def __call__(self, temp: torch.tensor,
                 free_probs: torch.Tensor,
                 induced_probs: torch.Tensor | None = None,
                 out_probs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Build full transition matrix.
        :param temp: Temperature(s).
        :param free_probs: Free relaxation speeds [..., N, N].
        :param induced_probs: Optional induced transitions [..., N, N].
        :param out_probs: Optional outgoing transition rates [..., N].
        :return: Transition matrix [..., N, N].

        Example (2-level system):

        Free relaxation (symmetric form):
            base_probs = [[0,  k'],
                          [k', 0]]

        Induced transitions:
            induced_probs = [[0,  i'],
                             [i', 0]]

        Outgoing rates:
            out_probs = [t, t]

        Resulting matrix:
            [[-2k' * exp(-(E2 - E1)/kT),   2k'],
             [ 2k' * exp(-(E2 - E1)/kT), -2k']] / (1 + exp(-(E2 - E1)/kT))

          + [[-i',  i'],
             [ i', -i']]

          - [[t, 0],
             [0, t]]
        """
        probs_matrix = self._probs_matrix(temp, free_probs)
        K = probs_matrix.shape[-1]
        indices = torch.arange(K, device=probs_matrix.device)
        probs_matrix[..., indices, indices] = -probs_matrix.sum(dim=-2)
        transition_matrix = probs_matrix

        if induced_probs is not None:
            induced_probs[..., indices, indices] = -induced_probs.sum(dim=-2)
            transition_matrix += induced_probs
        if out_probs is not None:
            transition_matrix -= torch.diag_embed(out_probs)
        return transition_matrix


class EvolutionMatrixKinetic(EvolutionMatrix):
    def __call__(self,
                 temp: torch.tensor,
                 free_probs: torch.Tensor,
                 induced_probs: torch.Tensor | None = None,
                 out_probs: torch.Tensor | None = None,
                 kynetic_probs: torch.Tensor | None = None):
        """
        :temp
        :param free_probs: The free relaxation speed. The shape of the __call__ is
        :param induced_probs:
        :param out_probs:
        :return:
        """
        probs_matrix = self._probs_matrix(temp, free_probs)
        K = probs_matrix.shape[-1]
        indices = torch.arange(K, device=probs_matrix.device)
        probs_matrix[..., indices, indices] = -probs_matrix.sum(dim=-2)
        transition_matrix = probs_matrix

        if induced_probs is not None:
            induced_probs[..., indices, indices] = -induced_probs.sum(dim=-2)
            transition_matrix += induced_probs
        if out_probs is not None:
            transition_matrix -= torch.diag_embed(out_probs)

        if kynetic_probs is not None:
            transition_matrix += kynetic_probs
        return transition_matrix


class EvolutionVectorSolver:
    @staticmethod
    def odeint_solver(time: torch.Tensor, initial_populations: torch.Tensor,
                     evo: EvolutionMatrix, matrix_generator: TransitionMatrixGenerator):
        def _rate_equation(t, n_flat, evo: EvolutionMatrix, matrix_generator: TransitionMatrixGenerator):
            """
            RHS for dn/dt = M(t) n, where M depends on t through temperature.
            - t: scalar time
            - n_flat: flattened populations of shape (..., K)
            Returns dn_flat/dt of same shape.
            """
            M_t = evo(*matrix_generator(t))
            dn = torch.matmul(M_t, n_flat.unsqueeze(-1)).squeeze(-1)
            return dn
        sol = odeint(func=lambda t, y: _rate_equation(
                     t, y, evo, matrix_generator),
                     y0=initial_populations,
                     t=time
                     )
        return sol

    @staticmethod
    def exponential_solver(time: torch.Tensor,
                          initial_populations: torch.Tensor,
                          evo: EvolutionMatrix, matrix_generator: TransitionMatrixGenerator,
                          chunk_size: tp.Optional[int] = None):
        dt = (time[1] - time[0])
        M = evo(*matrix_generator(time))
        exp_M = torch.matrix_exp(M * dt)

        size = time.size()[0]
        n = torch.zeros((size,) + initial_populations.shape, dtype=initial_populations.dtype)

        n[0] = initial_populations

        for i in range(len(time) - 1):
            current_n = n[i]  # Shape [..., K]
            next_n = torch.matmul(exp_M[i], current_n.unsqueeze(-1)).squeeze(-1)
            n[i + 1] = next_n
        return n

    @staticmethod
    def stationary_rate_solver(time: torch.Tensor,
                         initial_populations: torch.Tensor,
                         evo: EvolutionMatrix, matrix_generator: TransitionMatrixGenerator):
        M = evo(*matrix_generator(time[0]))

        dims_to_add = M.dim()

        reshape_dims = [len(time)] + [1] * dims_to_add
        time_reshaped = time.reshape(reshape_dims)

        exp_m = torch.matrix_exp(M * time_reshaped)
        n = torch.matmul(exp_m, initial_populations.unsqueeze(-1)).squeeze(-1)
        return n




