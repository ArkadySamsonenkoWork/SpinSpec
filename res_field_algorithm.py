import itertools
from itertools import chain
import warnings
from abc import ABC, abstractmethod
import typing as tp

import torch
from torch import nn

import spin_system


class ViewIndexator(nn.Module):
    def _is_increasing_sequence(self, tensor):
        if len(tensor) < 2:
            return True
        return torch.all(tensor[1:] - tensor[:-1] == 1)

    def single_indexation(self, indexes: torch.Tensor, args: tp.Sequence[torch.Tensor]):
        if len(indexes) == args[0].shape[0]:
            return args

        elif self._is_increasing_sequence(indexes):
            min_idx = indexes[0]
            max_idx = indexes[-1]
            return (arg[(min_idx-1):max_idx] for arg in args)

        else:
            return (arg.index_select(dim=0, index=indexes) for arg in args)

    def double_indexation(self, indexes_1: torch.Tensor, indexes_2: torch.Tensor, args: tp.Sequence[torch.Tensor]):
        idx = torch.cat((indexes_1, indexes_2), dim=0)
        n_1 = indexes_1.numel()
        reordered_array = self.single_indexation(idx, args)

        return (reordered[:n_1] for reordered in reordered_array)


class BaseEigenSolver(nn.Module, ABC):
    @abstractmethod
    def forward(self, F: torch.Tensor, G: torch.Tensor, B: torch.Tensor):
        """
        Compute only eigenvalues for H = F + G * B.
        :param F: Field-free Hamiltonian part, shape [..., K, K].
        :param G: Field-dependent Hamiltonian part, shape [..., K, K].
        :param B: Magnetic field at which to compute eigenvalues, shape [..., L].
        :return: Tuple of (eigenvalues, eigenvectors).
        """
        pass

    @abstractmethod
    def compute_eigenvalues(self, F: torch.Tensor, G: torch.Tensor, B: torch.Tensor):
        """
        Compute only eigenvalues for H = F + G * B.
        :param F: Field-free Hamiltonian part, shape [..., K, K].
        :param G: Field-dependent Hamiltonian part, shape [..., K, K].
        :param B: Magnetic field at which to compute eigenvalues, shape [..., L].
        :return: Eigenvalues.
        """
        pass

class EighEigenSolver(BaseEigenSolver):
    """
    Default eigen solver based on torch.linalg.eigh.
    """
    def forward(self, Hamiltonian: torch.Tensor):
        return torch.linalg.eigh(Hamiltonian)

    def compute_eigenvalues(self, F: torch.Tensor, G: torch.Tensor, B: torch.Tensor):
        return torch.linalg.eigvalsh(F + G * B)

# IT WAS BE REBUILDED
def has_sign_change(res_low: torch.Tensor, res_high: torch.Tensor) -> torch.Tensor:
    """
    calculate the criteria that delta_1N < resonance_frequency
    :param res_low: resonance function for the lower magnetic field in the interval. The shape is [..., num_pairs],
    where K is spin system dimension
    :param res_high: resonance function for the higher magnetic field in the interval. The shape is [..., num_pairs],
    where K is spin system dimension
    :return: mask with the shape [...]. If the value is True, the resonance function changes
    sign and segment can ve bisected further
    """
    mask = ((res_low * res_high) <= 0).any(dim=(-1))
    return mask


def has_rapid_variation(res_low: torch.Tensor, res_high: torch.Tensor,
                            deriv_max: torch.Tensor, B_low: torch.Tensor, B_high: torch.Tensor) -> torch.Tensor:
    """
    calculate the criteria that delta_1N < resonance_frequency
    :param res_low: resonance function for the lower magnetic field in the interval. The shape is [..., num_pairs],
    where K is spin system dimension
    :param res_high: resonance function for the higher magnetic field in the interval. The shape is [..., num_pairs],
    where K is spin system dimension
    :param deriv_max: It is a maxima of energy derevative En'(infinity).
    The calculations can be found in original article. The shape is [...]
    :param B_low: It is minima magnetic field of the interval. The shape is [..., 1, 1]
    :param B_high: It is maxima magnetic field of the interval. The shape is [..., 1, 1]
    :return: mask with the shape [...]. If the value is True, the segment could be bisected further.
    """
    threshold = (deriv_max * (B_high - B_low).squeeze(dim=(-2, -1))).unsqueeze(-1)
    mask = (((res_low + res_high) / 2).abs() <= threshold).any(dim=(-1))
    return mask


# Must me rebuild to speed up.
# 1) After each while iteration, it is possible to stack intervals to make bigger batches
# 2) Also, it is possible to stack all lists of tensor to one stack to increase the speed.
# 3) Maybe, it is better to avoid storing of deriv_max at the list and use indexes every time
# 4) converged_mask.any(). I have calculated the eigen_val and eigen_vec at the mid magnetic field.
# 5) Think about parallelogram point form the article. The resonance can not be excluded!!!!
# 6) Может, нужно всё NAN покрывать...
# 7) Возможно, где-то нужно добавить clone.
# 8) Возможно, стоит делить интервал не на две части, а искать точки разделения по полиному третьей степени.
# 9) Например, выбирать 10 точек и смотреть, где функция меняет знак.
# 10) Если дельта_1N < u0, то корень может быть только один. И резонансная функция меняет знак.
# Если дельта_1N >= u0, то корней может быть несколько, а может и не быть.
# But it doesn't mean that it must be.
# I can split the interval one more time. It can speed up the further calculations at next functions.
# 11) Нужно сделать один базовый класс и относледоваться от него. Разделить алгоритм на случай,
# когда baseline_sign всегда положительная или отрицательная
# 12) A + xB. Можно вынести все ядерные взаимодействия в отдельную матрицу и из-за этого ускорить вычисления.
# 13) При иттерации по батчам можно ввести распаралеливание на процессоре
# 14) Изменить способо обработки случая and. Сейчас там формируется два отдельных батча.
# 15) triu_indices - можно посчитать только один раз и потом не пересчитывать
# Можно ввести ещё одну размерность.

class BaseResonanceIntervalSolver(nn.Module, ABC):
    """
    Base class for algorithm of resonance interval search
    """
    def __init__(self, spin_dim: int,
                 eigen_finder: tp.Optional[BaseEigenSolver] = EighEigenSolver(), r_tol: float = 1e-5,
                 max_iterations: float = 20,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        super().__init__()
        self.eigen_finder = eigen_finder
        self.r_tol = torch.tensor(r_tol, dtype=dtype, device=device)
        self.max_iterations = torch.tensor(max_iterations)
        self.spin_dim = spin_dim
        self._triu_indices = torch.triu_indices(spin_dim, spin_dim, offset=1, device=device)
        self.view_indexator = ViewIndexator()

    def _compute_resonance_functions(self, eig_values_low: torch.Tensor, eig_values_high: torch.Tensor,
                                    resonance_frequency: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        calculate the resonance functions for eig_values.
        :param eig_values_low: energies in the ascending order at B_low magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_values_high: energies in the ascending order at B_high magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param resonance_frequency: resonance frequency. The shape is []
        :return: Resonance functions for left and right fields
        """

        u, v = self._triu_indices

        eig_values_high_expanded = eig_values_high.unsqueeze(-2)
        eig_values_low_expanded = eig_values_low.unsqueeze(-2)

        res_low = \
            (eig_values_low_expanded[..., v] - eig_values_low_expanded[..., u]).squeeze(
                -2) - resonance_frequency
        res_high = \
            (eig_values_high_expanded[..., v] - eig_values_high_expanded[..., u]).squeeze(
                -2) - resonance_frequency
        return res_low, res_high


    def _has_monotonically_rule(self, eig_values_low: torch.Tensor, eig_values_high: torch.Tensor,
                               resonance_frequency: torch.Tensor) -> torch.Tensor:
        """
        calculate the criteria that delta_1N < resonance_frequency
        :param eig_values_low: energies in the ascending order at B_low magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_values_high: energies in the ascending order at B_high magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param resonance_frequency: the resonance frequency. The shape is []
        :return:  mask with the shape [...]. If the value is True, the segment could be bisected further.
        """
        res_1N = eig_values_high[..., -1] - eig_values_high[..., 0] - resonance_frequency
        return res_1N >= 0

    def check_resonance(self, eig_values_low: torch.Tensor, eig_values_high: torch.Tensor,
                        B_low: torch.Tensor, B_high: torch.Tensor, resonance_frequency: torch.Tensor,
                        baseline_sign: tp.Optional[torch.Tensor], derivative_max: tp.Optional[torch.Tensor]):
        """
        Check the presence of the resonance at the interval for the general case. I
        :param eig_values_low: energies in the ascending order at B_low magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_values_high: energies in the ascending order at B_high magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param B_low: It is minima magnetic field of the interval. The shape is [..., 1, 1]
        :param B_high: It is maxima magnetic field of the interval. The shape is [..., 1, 1]
        It is needed to choose the test-criteria.  The shape is [...]
        :param resonance_frequency: The resonance frequency.
        :return: mask with the shape [...].  If it is true, the interval could be bisected further
        """
        mask_monotonically = self._has_monotonically_rule(
            eig_values_low, eig_values_high, resonance_frequency)  # [...]
        mask_loop_dependant = self.loop_dependant_mask(eig_values_low, eig_values_high, B_low, B_high,
                                                       resonance_frequency, baseline_sign, derivative_max)
        return torch.logical_and(mask_monotonically, mask_loop_dependant)

    @abstractmethod
    def loop_dependant_mask(self, eig_values_low, eig_values_high, B_low, B_high, resonance_frequency,
                            baseline_sign: tp.Optional[torch.Tensor], derivative_max: tp.Optional[torch.Tensor]):
        """
        Compute a mask based on loop-dependent resonance conditions.
        """
        pass

    def _compute_derivative(self, eigen_vector: torch.Tensor, G: torch.Tensor):
        """
        :param eigen_vector: eigen vectors of Hamiltonian
        :param G: Magnetic dependant part of the Hamiltonian: H = F + B * G
        :return: Derivatives of energies by magnetic field. The calculations are based on Feynman's theorem.
        """
        tmp = torch.matmul(G, eigen_vector)
        deriv = (torch.conj(eigen_vector) * tmp).sum(dim=-2).real
        return deriv

    def compute_error(self, eig_values_low: torch.Tensor, eig_values_mid: torch.Tensor,
                      eig_values_high: torch.Tensor,
                      eig_vectors_low: torch.Tensor,
                      eig_vectors_high: torch.Tensor,
                      B_low: torch.Tensor, B_high: torch.Tensor,
                      G: torch.Tensor,
                      row_indexes: torch.Tensor):
        """
        Compute the error after division of the interval
        :param eig_values_low: energies in the ascending order at B_low magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_values_mid: energies in the ascending order at B_mid magnetic field.
        The shape is [..., K], where K is spin system dimension. B_mid = (B_low + B_high) / 2
        :param eig_values_high: energies in the ascending order at B_high magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_vectors_low: eigen vectors corresponding eig_values_low. The shape is [..., K, K],
        where K is spin system dimension
        :param eig_vectors_high: eigen vectors corresponding eig_values_high. The shape is [..., K, K],
        where K is spin system dimension
        :param B_low: The lower magnetic field The shape is [..., 1, 1]
        :param B_high: The higher magnetic field The shape is [..., 1, 1]
        :param row_indexes: Indexes where Gz must be sliced. The long-indexes tensor. The shape == reduced number of  indexes
        :param G: The magnetic field dependant part of the Hamiltonian: F + G * B. The shape is [..., K, K]
        :return: epsilon is epsilon mistake. The tensor with the shape [...]
        """
        G_idx = G.index_select(0, row_indexes)
        derivatives_low = self._compute_derivative(eig_vectors_low, G_idx)
        derivatives_high = self._compute_derivative(eig_vectors_high, G_idx)
        eig_values_estimation = 0.5 * (eig_values_high + eig_values_low) +\
                                      (B_high - B_low) / 8 * (derivatives_high - derivatives_low)

        epsilon = 2 * (eig_values_estimation - eig_values_mid).abs().max(dim=-1)[0]
        return epsilon, (derivatives_low, derivatives_high)

    @abstractmethod
    def determine_split_masks(self, eig_values_low, eig_values_mid, eig_values_high,
                              B_low, B_mid, B_high, row_indexes,
                              resonance_frequency, baseline_sign, derivative_max):
        pass

    def assemble_current_batches(self,
                     eig_values_low, eig_values_mid, eig_values_high,
                     eig_vectors_low, eig_vectors_mid, eig_vectors_high,
                     B_low, B_mid, B_high, row_indexes,
                     resonance_frequency,
                     baseline_sign: tp.Optional[torch.Tensor], derivative_max: tp.Optional[torch.Tensor]
                                 ):
        new_intervals = []

        mask_left, mask_right = self.determine_split_masks(eig_values_low, eig_values_mid, eig_values_high,
                                                    B_low, B_mid, B_high,
                                                    row_indexes, resonance_frequency, baseline_sign, derivative_max)

        mask_and = torch.logical_and(mask_left, mask_right)
        mask_xor = torch.logical_xor(mask_left, mask_right)
        # Process and case. It means that both intervals have resonance
        if mask_and.any():
            indexes_and = mask_and.nonzero(as_tuple=True)[0]
            raw_indexes_and = row_indexes.clone()[indexes_and]


            new_intervals.append(
                ((eig_values_low.index_select(0, indexes_and), eig_values_mid.index_select(0, indexes_and)),
                 (eig_vectors_low.index_select(0, indexes_and), eig_vectors_mid.index_select(0, indexes_and)),
                 (B_low.index_select(0, indexes_and), B_mid.index_select(0, indexes_and)),
                 raw_indexes_and)
            )
            new_intervals.append(
                ((eig_values_mid.index_select(0, indexes_and), eig_values_high.index_select(0, indexes_and)),
                 (eig_vectors_mid.index_select(0, indexes_and), eig_vectors_high.index_select(0, indexes_and)),
                 (B_mid.index_select(0, indexes_and), B_high.index_select(0, indexes_and)),
                 raw_indexes_and)
            )
        # Process XOR case. It means that only one interval has resonance.
        # Note, that it is impossible that none interval has resonance
        if mask_xor.any():
            idx_xor = mask_xor.nonzero(as_tuple=True)[0]
            mask_left = mask_left.index_select(0, idx_xor)
            new_intervals.append(
                self._compute_xor_interval(
                    idx_xor,
                    mask_left,
                    mask_right,
                    eig_values_low, eig_values_mid, eig_values_high,
                    eig_vectors_low, eig_vectors_mid, eig_vectors_high,
                    B_low, B_mid, B_high,
                    row_indexes,
                )
            )
        return new_intervals

    def _single_index_data(self, indexes: torch.Tensor, args):
        if args[0].shape[0] == len(indexes):
            return args
        else:
            return (arg.index_select(dim=0, index=indexes) for arg in args)

    def _compute_xor_interval(self,
            idx_xor: torch.Tensor,
            mask_left: torch.Tensor,
            mask_right: torch.Tensor,
            eig_values_low, eig_values_mid, eig_values_high,
            eig_vectors_low, eig_vectors_mid, eig_vectors_high,
            B_low, B_mid, B_high,
            row_indexes: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        :param idx_xor: the xor indexes of the left and right mask
         It means that resonance happens in the one interval left or right. The shape is [...]
        :param mask_left: the left mask with the values in batches where resonance happens. The shape is [reduced number of  indexes]
        :param mask_right: the right mask with the values in batches where resonance happens. The shape is [...]
        :param eig_values_low: energies in the ascending order at B_low magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_values_mid: energies in the ascending order at B_mid magnetic field.
        The shape is [..., K], where K is spin system dimension. B_mid = (B_low + B_high) / 2
        :param eig_values_high: energies in the ascending order at B_high magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_vectors_low: eigen vectors corresponding eig_values_low. The shape is [..., K, K],
        where K is spin system dimension
        :param eig_vectors_mid: eigen vectors corresponding eig_values_mid. The shape is [..., K, K],
        where K is spin system dimension
        :param eig_vectors_high: eigen vectors corresponding eig_values_high. The shape is [..., K, K],
        where K is spin system dimension
        :param B_low: The lower magnetic field The shape is [..., 1, 1]
        :param B_mid: The middel magnetic field The shape is [..., 1, 1]
        :param B_high: The high magnetic field The shape is [..., 1, 1]
        :param row_indexes: Indexes where Gz must be sliced. The long-indexes tensor. The shape == reduced number of  indexes
        :return: tuple of eig_values, eig_vectors, magnetic fields, and new indexes
        """
        if B_low.shape[0] == len(idx_xor):
            pass

        else:
            B_low = B_low.index_select(dim=0, index=idx_xor)
            B_mid = B_mid.index_select(dim=0, index=idx_xor)
            B_high = B_high.index_select(dim=0, index=idx_xor)

            eig_values_low = eig_values_low.index_select(dim=0, index=idx_xor)
            eig_values_mid = eig_values_mid.index_select(dim=0, index=idx_xor)
            eig_values_high = eig_values_high.index_select(dim=0, index=idx_xor)

            eig_vectors_low = eig_vectors_low.index_select(dim=0, index=idx_xor)
            eig_vectors_mid = eig_vectors_mid.index_select(dim=0, index=idx_xor)
            eig_vectors_high = eig_vectors_high.index_select(dim=0, index=idx_xor)


        torch.where(mask_left.unsqueeze(-1).unsqueeze(-1), B_low, B_mid, out=B_low)
        torch.where(mask_left.unsqueeze(-1).unsqueeze(-1), B_mid, B_high, out=B_high)

        torch.where(mask_left.unsqueeze(-1), eig_values_low, eig_values_mid, out=eig_values_low)
        torch.where(mask_left.unsqueeze(-1), eig_values_mid, eig_values_high, out=eig_values_high)

        torch.where(mask_left.unsqueeze(-1).unsqueeze(-1), eig_vectors_low, eig_vectors_mid, out=eig_vectors_low)
        torch.where(mask_left.unsqueeze(-1).unsqueeze(-1), eig_vectors_mid, eig_vectors_high, out=eig_vectors_high)


        row_indexes = row_indexes.clone().index_select(0, idx_xor)

        return (
            (eig_values_low, eig_values_high),
            (eig_vectors_low, eig_vectors_high),
            (B_low, B_high),
            row_indexes
        )

    def _compute_xor_interval_dirivative(self,
                              idx_xor: torch.Tensor,
                              mask_left: torch.Tensor,
                              mask_right: torch.Tensor,
                              eig_values_low, eig_values_mid, eig_values_high,
                              eig_vectors_low, eig_vectors_mid, eig_vectors_high,
                              derivatives_low, derivatives_mid, derivatives_high,
                              B_low, B_mid, B_high,
                              row_indexes: torch.Tensor
                              ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor], torch.Tensor
    ]:
        mask_left = mask_left.index_select(0, idx_xor)
        (
         (eig_values_low, eig_values_high),
         (eig_vectors_low, eig_vectors_high),
         (B_low, B_high), row_indexes
        ) = self._compute_xor_interval(
                    idx_xor,
                    mask_left,
                    mask_right,
                    eig_values_low, eig_values_mid, eig_values_high,
                    eig_vectors_low, eig_vectors_mid, eig_vectors_high,
                    B_low, B_mid, B_high,
                    row_indexes,
                )
        if derivatives_low.shape[0] == len(idx_xor):
            pass
        else:
            derivatives_low = derivatives_low.index_select(dim=0, index=idx_xor)
            derivatives_mid = derivatives_mid.index_select(dim=0, index=idx_xor)
            derivatives_high = derivatives_high.index_select(dim=0, index=idx_xor)

        torch.where(mask_left.unsqueeze(1), derivatives_low, derivatives_mid, out=derivatives_low)
        torch.where(mask_left.unsqueeze(1), derivatives_mid, derivatives_high, out=derivatives_high)

        return (
            (eig_values_low, eig_values_high),
            (eig_vectors_low, eig_vectors_high),
            (derivatives_low, derivatives_high),
            (B_low, B_high),
            row_indexes
        )

    def _finilize_batch(self,
                        eig_values_low: torch.Tensor, eig_values_mid: torch.Tensor, eig_values_high: torch.Tensor,
                        eig_vectors_low: torch.Tensor, eig_vectors_mid: torch.Tensor, eig_vectors_high: torch.Tensor,
                        derivatives_low: torch.Tensor, derivatives_mid: torch.Tensor, derivatives_high: torch.Tensor,
                        B_low: torch.Tensor, B_mid: torch.Tensor, B_high: torch.Tensor, row_indexes: torch.Tensor,
                        resonance_frequency: torch.Tensor, baseline_sign: tp.Optional[torch.Tensor],
                        derivative_max: tp.Optional[torch.Tensor]):

        mask_left, mask_right = self.determine_split_masks(eig_values_low, eig_values_mid, eig_values_high,
                                                    B_low, B_mid, B_high,
                                                    row_indexes, resonance_frequency, baseline_sign, derivative_max)

        mask_and = torch.logical_and(mask_left, mask_right)
        mask_xor = torch.logical_xor(mask_left, mask_right)
        new_intervals = []
        if mask_and.any():
            indexes_and = mask_and.nonzero(as_tuple=True)[0]
            raw_indexes_and = row_indexes.clone().index_select(0, indexes_and)
            new_intervals.append(
                ((eig_values_low.index_select(0, indexes_and), eig_values_mid.index_select(0, indexes_and)),
                 (eig_vectors_low.index_select(0, indexes_and), eig_vectors_mid.index_select(0, indexes_and)),
                 (derivatives_low.index_select(0, indexes_and), derivatives_mid.index_select(0, indexes_and)),
                 (B_low.index_select(0, indexes_and), B_mid.index_select(0, indexes_and)),
                 raw_indexes_and)
            )
            new_intervals.append(
                ((eig_values_mid.index_select(0, indexes_and), eig_values_high.index_select(0, indexes_and)),
                 (eig_vectors_mid.index_select(0, indexes_and), eig_vectors_high.index_select(0, indexes_and)),
                 (derivatives_mid.index_select(0, indexes_and), derivatives_high.index_select(0, indexes_and)),
                 (B_mid.index_select(0, indexes_and), B_high.index_select(0, indexes_and)),
                 raw_indexes_and)
            )
        # Process XOR case. It means that only one interval has resonance.
        # Note, that it is impossible that none interval has resonance
        if mask_xor.any():
            idx_xor = mask_xor.nonzero(as_tuple=True)[0]
            new_intervals.append(
                self._compute_xor_interval_dirivative(
                    idx_xor,
                    mask_left,
                    mask_right,
                    eig_values_low, eig_values_mid, eig_values_high,
                    eig_vectors_low, eig_vectors_mid, eig_vectors_high,
                    derivatives_low, derivatives_mid, derivatives_high,
                    B_low, B_mid, B_high,
                    row_indexes,
                )
            )
        return new_intervals

    def _iterate_batch(self, batch: tuple[
                                 tuple[torch.Tensor, torch.Tensor],
                                 tuple[torch.Tensor, torch.Tensor],
                                 tuple[torch.Tensor, torch.Tensor],
                                 torch.Tensor],
                        F: torch.Tensor, Gz: torch.Tensor,
                        resonance_frequency: torch.Tensor, a_tol: torch.Tensor,
                        baseline_sign: tp.Optional[torch.Tensor],
                        derivative_max: tp.Optional[torch.Tensor]):
        """
        :param batch: tuple of next values: (eig_values_low, eig_values_high),
        (eig_vectors_low, eig_vectors_high), (B_low, B_high), indexes
        :param F:
        :param Gz:
        :param resonance_frequency:
        :param a_tol:
        :param args:
        :param kwargs:
        :return: tuple of two lists: new batches for iteration and final batches for further processing
        """
        final_batches = []
        (eig_values_low, eig_values_high), (eig_vectors_low, eig_vectors_high), (B_low, B_high), row_indexes = batch
        B_mid = (B_low + B_high) / 2
        eig_values_mid, eig_vectors_mid = self.eigen_finder(
            F.index_select(0, row_indexes) + Gz.index_select(0, row_indexes) * B_mid
        )
        # It is only one    single
        # point where gradient should be calculated
        error, (derivatives_low, derivatives_high) = \
            self.compute_error(eig_values_low, eig_values_mid, eig_values_high,
                               eig_vectors_low, eig_vectors_high,
                               B_low, B_high, Gz, row_indexes
                               )
        converged_mask = (error <= a_tol).any(dim=-1)

        active_mask = ~converged_mask

        converged_idx = converged_mask.nonzero(as_tuple=True)[0]
        active_idx = active_mask.nonzero(as_tuple=True)[0]
        if converged_mask.any():
            row_indexes_conv = row_indexes.clone()[converged_idx]
            # indexes_conv[indexes_conv == True] = converged_mask

            eig_vectors_mid_converg = eig_vectors_mid.index_select(0, converged_idx)
            derivatives_mid = self._compute_derivative(eig_vectors_mid_converg, Gz.index_select(0, row_indexes_conv))
            final_batches.extend(
                self._finilize_batch(
                    eig_values_low.index_select(0, converged_idx), eig_values_mid.index_select(0, converged_idx),
                    eig_values_high.index_select(0, converged_idx),
                    eig_vectors_low.index_select(0, converged_idx), eig_vectors_mid_converg,
                    eig_vectors_high.index_select(0, converged_idx),
                    derivatives_low.index_select(0, converged_idx), derivatives_mid,
                    derivatives_high.index_select(0, converged_idx),
                    B_low.index_select(0, converged_idx), B_mid.index_select(0, converged_idx),
                    B_high.index_select(0, converged_idx),
                    row_indexes_conv, resonance_frequency, baseline_sign, derivative_max)
            )
            """
            final_batches.append((
                (eig_values_low[converged_mask], eig_values_high[converged_mask]),
                (eig_vectors_low[converged_mask], eig_vectors_high[converged_mask]),
                (derivatives_low[converged_mask], derivatives_high[converged_mask]),
                (B_low[converged_mask], B_high[converged_mask]),
                indexes_conv,
            ))
            """
        if not active_mask.any():
            return [], final_batches

        # Update active components.
        B_low = B_low.index_select(0, active_idx)
        B_high = B_high.index_select(0, active_idx)
        B_mid = B_mid.index_select(0, active_idx)

        eig_values_low = eig_values_low.index_select(0, active_idx)
        eig_values_mid = eig_values_mid.index_select(0, active_idx)
        eig_values_high = eig_values_high.index_select(0, active_idx)

        eig_vectors_low = eig_vectors_low.index_select(0, active_idx)
        eig_vectors_mid = eig_vectors_mid.index_select(0, active_idx)
        eig_vectors_high = eig_vectors_high.index_select(0, active_idx)
        # indexes = batch["indexes"]

        new_batches = self.assemble_current_batches(
            eig_values_low, eig_values_mid, eig_values_high,
            eig_vectors_low, eig_vectors_mid, eig_vectors_high,
            B_low, B_mid, B_high, row_indexes.index_select(0, active_idx), resonance_frequency,
            baseline_sign, derivative_max)
        return new_batches, final_batches

    # Вероятно, нужно будет поменять на дикты. Но будут проблемы с jit-компиляцией
    def forward(self, F: torch.Tensor, Gz: torch.Tensor,
                                B_low: torch.Tensor, B_high: torch.Tensor,
                                resonance_frequency: torch.Tensor,
                baseline_sign: tp.Optional[torch.Tensor], derivative_max: tp.Optional[torch.Tensor]) ->\
            list[tuple[tuple[torch.Tensor, torch.Tensor],
                       tuple[torch.Tensor, torch.Tensor],
                       tuple[torch.Tensor, torch.Tensor],
                       tuple[torch.Tensor, torch.Tensor],
                       torch.Tensor]
            ]:
        """
        Calculate the resonance intervals, where the resonance field is possible.
        :param F: Magnetic filed free stationary Hamiltonian matrix. The shape is [..., K, K],
        where K is spin system dimension
        :param Gz: Magnetic field dependant part of stationary Hamiltonian with the shape [..., K, K].
        :param B_low: The start of the interval to find roots. The shape is [...]
        :param B_high: The end of the interval to find roots. The shape is [...]
        :param resonance_frequency: The resonance frequency. The shape is []
        :return: list of tuples. Each tuple it is the parameters of the interval:
             (eig_values_low, eig_values_high) - eigen values of Hamiltonian
             at the low and high magnetic fields
             (eig_vectors_low, eig_vectors_high) - eigen vectors of Hamiltonian
             at the low and high magnetic fields
             (energy_derivatives_low, energy_derivatives_high) - the derivatives of the energy at
             low and high magnetic field
             (energy_derivatives_low, energy_derivatives_high) - the derivatives of the energy
             at low and high magnetic field
             (energy_derivatives_low, energy_derivatives_high) - the derivatives of the energy
             at low and high magnetic field
             indexes, where mask is valid

        """
        a_tol = resonance_frequency * self.r_tol
        B_low = B_low[..., None, None]
        B_high = B_high[..., None, None]

        Hamiltonians = torch.stack((F + Gz * B_low, F + Gz * B_high), dim=-3)
        eig_values, eig_vectors = self.eigen_finder(Hamiltonians)
        eig_values_low, eig_values_high = eig_values[..., 0, :], eig_values[..., 1, :]
        eig_vectors_low, eig_vectors_high = eig_vectors[..., 0, :, :], eig_vectors[..., 1, :, :]

        iterations = 0

        active_mask = self.check_resonance(eig_values_low, eig_values_high,
                                           B_low, B_high, resonance_frequency, baseline_sign, derivative_max
                                           )
        raw_indexes = active_mask.nonzero(as_tuple=True)[0]
        if torch.all(~active_mask):
            warnings.warn("There are no resonance in the interval")
        final_batches = []
        current_batches = [(
            (eig_values_low[raw_indexes], eig_values_high[raw_indexes]),
            (eig_vectors_low[raw_indexes], eig_vectors_high[raw_indexes]),
            (B_low[raw_indexes], B_high[raw_indexes]),
            raw_indexes
        )]

        while current_batches:
            iteration_results = [
                self._iterate_batch(batch, F, Gz, resonance_frequency, a_tol, baseline_sign, derivative_max)
                for batch in current_batches]
            current_batches = [current_batch for batches in iteration_results for current_batch in batches[0]]
            final_batches.extend([current_batch for batches in iteration_results for current_batch in batches[1]])
            iterations += 1

        # locate_resonance_fields(final_batches, resonance_frequency)
        return final_batches


class GeneralResonanceIntervalSolver(BaseResonanceIntervalSolver):
    """
    Find resonance interval for the general Hamiltonian case.
    The general case determines form the conditcion:
        delta_1N. If it is greater nu_0, than looping resonance are possible. If not, the resonance interval
        can be determined by change sign of resonance function at the ends of the interval.
    It is used if for part of the data among the beach-mesh dimension, the conditions are True, and for part are False
    """

    def loop_dependant_mask(self, eig_values_low: torch.Tensor, eig_values_high: torch.Tensor,
                        B_low: torch.Tensor, B_high: torch.Tensor,
                        resonance_frequency: torch.Tensor, baseline_sign_mask: torch.Tensor, deriv_max: torch.Tensor,
                        ):
        """
        Check the mask depending on the presence of the looping resonance.
        :param eig_values_low: energies in the ascending order at B_low magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_values_high: energies in the ascending order at B_high magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param B_low: It is minima magnetic field of the interval. The shape is [..., 1, 1]
        :param B_high: It is maxima magnetic field of the interval. The shape is [..., 1, 1]
        :param deriv_max: The maximum value of the energy derivatives. The shape is [...]
        :param baseline_sign_mask: The mask that shows the behaviour of the delta_1N at zero field.
        It is needed to choose the test-criteria.  The shape is [...]
        :param resonance_frequency: The resonance frequency.
        :return: mask with the shape [...].  If it is true, the interval could be bisected further
        """
        res_low, res_high = self._compute_resonance_functions(
            eig_values_low, eig_values_high, resonance_frequency)
        mask = has_rapid_variation(res_low, res_high, deriv_max, B_low, B_high)
        mask_sign_change = has_sign_change(res_low, res_high)
        torch.where(baseline_sign_mask, mask, mask_sign_change, out=mask)

        return mask

    def determine_split_masks(self, eig_values_low, eig_values_mid, eig_values_high,
                                                    B_low, B_mid, B_high,
                                                    row_indexes, resonance_frequency, baseline_sign, deriv_max):
        baseline_sign_idx = baseline_sign.index_select(0, row_indexes)
        deriv_max_idx = deriv_max.index_select(0, row_indexes)

        mask_left = self.check_resonance(eig_values_low, eig_values_mid, B_low, B_mid,
                                        resonance_frequency, baseline_sign_idx, deriv_max_idx)
        mask_right = self.check_resonance(eig_values_mid, eig_values_high, B_mid, B_high,
                                         resonance_frequency, baseline_sign_idx, deriv_max_idx)

        return mask_left, mask_right


class ZeroFreeResonanceIntervalSolver(BaseResonanceIntervalSolver):
    """
    Find the resonance intervals for the case when delta_1N < nu_o. For this case the looping resonance is impossible.
    The general case determines form the condition:
        delta_1N. If it is greater nu_0, than looping resonance are possible. If not, the resonance interval
        can be determined by change sign of resonance function at the ends of the interval.
    """

    def loop_dependant_mask(self, eig_values_low: torch.Tensor, eig_values_high: torch.Tensor,
                        B_low: torch.Tensor, B_high: torch.Tensor,
                        resonance_frequency: torch.Tensor, baseline_sign, derivative_max
                        ):
        """
        Check the mask depending on the presence of the looping resonance.
        :param eig_values_low: energies in the ascending order at B_low magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_values_high: energies in the ascending order at B_high magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param B_low: It is minima magnetic field of the interval. The shape is [..., 1, 1]
        :param B_high: It is maxima magnetic field of the interval. The shape is [..., 1, 1]
        :param resonance_frequency: The resonance frequency.
        :return: mask with the shape [...].  If it is true, the interval could be bisected further
        """
        res_low, res_high = self._compute_resonance_functions(
            eig_values_low, eig_values_high, resonance_frequency)
        mask_sign_change = has_sign_change(res_low, res_high)
        return mask_sign_change

    def determine_split_masks(self, eig_values_low, eig_values_mid, eig_values_high,
                            B_low, B_mid, B_high, row_indexes, resonance_frequency, baseline_sign, derivative_max):
        mask_left = self.check_resonance(eig_values_low, eig_values_mid, B_low, B_mid,
                                         resonance_frequency, baseline_sign, derivative_max)
        mask_right = self.check_resonance(eig_values_mid, eig_values_high, B_mid, B_high,
                                          resonance_frequency, baseline_sign, derivative_max)
        return mask_left, mask_right


# Может это нужно делать через разреженные матрицы.... Я не понимаю...
# Я считаю все произведения U Gx V даже, если переходов нет. Если переходов нет, то вес ноль. Может,
# Это стоит оптимизировать и сделать опреацию дешевле и считать только в valid_u и valid_b.
# Но тогда стоятся лишние струкутуры
# Может быть стоит разделять батчи дальше по u и v....
# Дважды считаю коэффициенты полинома. Нужно будет переделать. Один раз для нахождения корней,
# Один раз для получения энергий.
# Возможно, стоит избавиться от двух масок.


class BaseResonanceLocator(nn.Module):
    def __init__(self, max_iterations=50, tolerance=1e-12, accuracy=1e-5, output_full_eigenvector=False,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.max_iterations_newton = torch.tensor(max_iterations, device=device)
        self.tolerance_newton = torch.tensor(tolerance, device=device, dtype=dtype)
        self.accuracy_newton = torch.tensor(accuracy, device=device, dtype=dtype)
        self.output_full_eigenvector = output_full_eigenvector

    def _compute_cubic_polinomial_coeffs(self,
                                         diff_eig_low: torch.Tensor,
                                         diff_eig_high: torch.Tensor,
                                         diff_deriv_low: torch.Tensor,
                                         diff_deriv_high: torch.Tensor):


        deriv_sum = diff_deriv_low + diff_deriv_high

        coef_3 = 2 * (diff_eig_low - diff_eig_high) + deriv_sum
        coef_2 = -3 * (diff_eig_low - diff_eig_high) - 2 * diff_deriv_low - diff_deriv_high
        coef_1 = diff_deriv_low
        coef_0 = diff_eig_low
        return (coef_3, coef_2, coef_1, coef_0)

    def _find_one_root_newton(
            self, coef_3: torch.Tensor, coef_2: torch.Tensor, coef_1: torch.Tensor, coef_0: torch.Tensor):
        """
        Finds the solution of the equation:
        coef_3 * t^3 + coef_2 * t^2 + coef_1 * t + coef_0 = 0.

        It works when there is only one singe root
        :param coef_3: coefficient at t^3
        :param coef_2: coefficient at t^2
        :param coef_1: coefficient at t^1
        :param coef_0: coefficient at t^0
        :return: The root od the polynomial
        """
        t = - coef_0 / (coef_1 + coef_2 + coef_3)

        for _ in range(self.max_iterations_newton):
            poly_val = coef_3 * t ** 3 + coef_2 * t ** 2 + coef_1 * t + coef_0
            poly_deriv = 3 * coef_3 * t ** 2 + 2 * coef_2 * t + coef_1
            delta = poly_val / (poly_deriv + self.tolerance_newton)
            t -= delta
            if (delta.abs() < self.accuracy_newton).all():
                break
        return t

    def _find_resonance_roots(self, diff_eig_low: torch.Tensor,
                                    diff_eig_high: torch.Tensor,
                                    diff_deriv_low: torch.Tensor,
                                    diff_deriv_high: torch.Tensor,
                                    resonance_frequency: torch.Tensor):
        """
        Finds the magnetic field value (B_mid) where a resonance occurs by solving a cubic equation
        using the Newton–Raphson method. It suggested that the resonance point is just one

        The cubic polynomial is defined by:
            p3 * t^3 + p2 * t^2 + p1 * t + p0 = 0
        with coefficients constructed from the input parameters:
            p3 = 2 * diff_eig_low - 2 * diff_eig_high + diff_deriv_low + diff_deriv_high
            p2 = -3 * diff_eig_low + 3 * diff_eig_high - 2 * diff_deriv_low - diff_deriv_high
            p1 = diff_deriv_low
            p0 = diff_eig_low - target_resonance
        :param diff_eig_low: Difference of eigenvalues at B_min for the pair, shape compatible with u and v.
        :param diff_eig_high: Difference of eigenvalues at B_max for the pair.
        :param diff_deriv_low: Difference of derivatives at B_min for the pair.
        :param diff_deriv_high: Difference of derivatives at B_max for the pair.
        :param resonance_frequency: The resonance frequency (or energy) to be reached.
        :return: Estimated magnetic field values where resonance occurs, shape matching input pair dimensions.
        """

        (coef_3, coef_2, coef_1, coef_0) = self._compute_cubic_polinomial_coeffs(
            diff_eig_low, diff_eig_high, diff_deriv_low, diff_deriv_high)
        coef_0 -= resonance_frequency

        return self._find_one_root_newton(coef_3, coef_2, coef_1, coef_0)

    def get_resonance_mask(self, diff_eig_low: torch.Tensor, diff_eig_high: torch.Tensor,
                           B_low: torch.Tensor, B_high: torch.Tensor,
                           resonance_frequency: torch.Tensor, indexes: torch.Tensor,
                           baseline_sign: tp.Optional[torch.Tensor], derivative_max: tp.Optional[torch.Tensor]):
        sign_change_mask = ((diff_eig_low - resonance_frequency) * (diff_eig_high - resonance_frequency) <= 0)
        return sign_change_mask

    def _compute_linear_interpolation_weights(self, step_B):
        """
        :param step_B:
        :return:
        """
        weights_low = step_B.unsqueeze(-1)
        weights_high = (1 - step_B).unsqueeze(-1)
        return weights_low, weights_high

    def _compute_resonance_fields(self, diff_eig_low: torch.Tensor,
                                  diff_eig_high: torch.Tensor,
                                  diff_deriv_low: torch.Tensor,
                                  diff_deriv_high: torch.Tensor,
                                  mask_trans: torch.Tensor,
                                  resonance_frequency: torch.Tensor) ->\
            list[tuple[torch.Tensor, torch.Tensor]]:
        step_B = torch.zeros_like(mask_trans, dtype=resonance_frequency.dtype)
        step_B[mask_trans] = self._find_resonance_roots(diff_eig_low, diff_eig_high,
                                                  diff_deriv_low, diff_deriv_high,
                                                  resonance_frequency)
        return [(step_B, mask_trans)]

    def _compute_resonance_energies(self,
                                    step_B: torch.Tensor,
                                    diff_eig_low: torch.Tensor,
                                    diff_eig_high: torch.Tensor,
                                    diff_deriv_low: torch.Tensor,
                                    diff_deriv_high: torch.Tensor
                                    ):
        step_B = step_B.unsqueeze(-1)
        (coef_3, coef_2, coef_1, coef_0) = self._compute_cubic_polinomial_coeffs(
            diff_eig_low, diff_eig_high, diff_deriv_low, diff_deriv_high)
        energy = coef_3 * step_B ** 3 + coef_2 * step_B ** 2 + coef_1 * step_B + coef_0
        return energy

    def _interpolate_vectors(self, vec_low, vec_high, weights_low, weights_high, eps=1e-12):
        """
        Differences / improvements:
        - Avoids computing a division for entries where the inner product is (near) zero by using a boolean mask
          and doing the division only for true entries (reduces work and avoids creating large temporaries).
        - Uses vec.abs().square().sum(...).sqrt() for norm which is usually cheaper than conj()*sum or linalg.norm.
        - Minimizes number of intermediate tensors.
        - Keeps the same numerical handling for tiny magnitudes (controlled by eps).
        """
        inner = torch.sum(vec_low.conj() * vec_high, dim=-1, keepdim=True)
        abs_inner = inner.abs()

        phase = torch.ones_like(inner)
        mask = abs_inner > eps
        if mask.any():
            phase[mask] = inner[mask] / abs_inner[mask]

        vec_high_aligned = vec_high * phase.conj()
        vec = vec_low * weights_low + vec_high_aligned * weights_high

        return vec

    def _split_mask(self, mask_res: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param mask_res: Boolean tensor of shape [batch_size, num_transitions],
                where each element indicates whether a specific transition occurred in that batch item.

        :return: tuple[torch.Tensor, torch.Tensor]:
                - mask_triu (torch.Tensor): Boolean vector of length num_transitions.
                  mask_triu[j] is True if transition j occurred in any batch element.
                - mask_trans (torch.Tensor): Subset of mask_res selecting only the columns
                  where mask_triu is True (i.e., transitions that occur at least once in the batch).
        """
        mask_triu = mask_res.any(dim=-2)
        mask_trans = mask_res[..., mask_triu]
        return mask_triu, mask_trans

    def _prepare_inputs(self, batch: tuple[
                                     tuple[torch.Tensor, torch.Tensor],
                                     tuple[torch.Tensor, torch.Tensor],
                                     tuple[torch.Tensor, torch.Tensor],
                                     tuple[torch.Tensor, torch.Tensor],
                                     torch.Tensor]
                        ):
        (eig_low, eig_high), (vec_low, vec_high), (deriv_low, deriv_high), (B_low, B_high), indexes = batch


        delta_B = B_high - B_low
        K = eig_low.shape[-1]
        lvl_down, lvl_up = torch.triu_indices(K, K, offset=1, device=B_low.device)

        deriv_low = deriv_low.unsqueeze(-2)
        deriv_high = deriv_high.unsqueeze(-2)
        eig_low = eig_low.unsqueeze(-2)
        eig_high = eig_high.unsqueeze(-2)

        return eig_low, eig_high, vec_low, vec_high, deriv_low, deriv_high,\
            B_low, B_high, indexes, delta_B, lvl_down, lvl_up

    def _compute_raw_differences(
        self, eig_low: torch.Tensor, eig_high: torch.Tensor, deriv_low: torch.Tensor,
            deriv_high: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor):

        diff_low = (eig_low[..., lvl_up] - eig_low[..., lvl_down]).squeeze(-2)
        diff_high = (eig_high[..., lvl_up] - eig_high[..., lvl_down]).squeeze(-2)
        raw_d_low = (deriv_low[..., lvl_up] - deriv_low[..., lvl_down])
        raw_d_high = (deriv_high[..., lvl_up] - deriv_high[..., lvl_down])
        return diff_low, diff_high, raw_d_low, raw_d_high

    def _compute_eigenvector_full_system(
            self,
            eig_vectors_low: torch.Tensor, eig_vectors_high: torch.Tensor,
            lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            weights_low: torch.Tensor, weights_high: torch.Tensor) -> torch.Tensor:
        """
        :param eig_vectors_low:
        :param eig_vectors_high:
        :param lvl_down:
        :param lvl_up:
        :param step_B:
        :return: eigen vectors of all states for all transitions.
        The shape is [..., num_transitions, K, K], where K is spin system dimension.
        The last dimension is stated. This is common agreement as in torch or numpy
        """
        low = eig_vectors_low.unsqueeze(-3).transpose(-1, -2)
        high = eig_vectors_high.unsqueeze(-3).transpose(-1, -2)

        weights_low = weights_low.unsqueeze(-1)
        weights_high = weights_high.unsqueeze(-1)

        eigen_vectors_system = self._interpolate_vectors(low, high, weights_low, weights_high).transpose(-1, -2)
        return eigen_vectors_system

    def _compute_eigenvectors_transitions(
            self,
            eig_vectors_low: torch.Tensor, eig_vectors_high: torch.Tensor,
            lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            weights_low: torch.Tensor, weights_high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param eig_vectors_low:
        :param eig_vectors_high:
        :param lvl_down:
        :param lvl_up:
        :param step_B:
        :return: Eigen vectors only states that includes in transitions.
        The shape is [..., num_transitions, K, ], where K is spin system dimension
        The first tensor is low states,
        the second vector is high states
        """
        # Be carefully with the order (:, lvl_down) vs (lvl_down, :)
        low_down = eig_vectors_low[..., :, lvl_down].transpose(-1, -2)
        high_down = eig_vectors_high[..., :, lvl_down].transpose(-1, -2)
        low_up = eig_vectors_low[..., :, lvl_up].transpose(-1, -2)
        high_up = eig_vectors_high[..., :, lvl_up].transpose(-1, -2)

        vectors_u = self._interpolate_vectors(low_down, high_down, weights_low, weights_high)
        vectors_v = self._interpolate_vectors(low_up, high_up, weights_low, weights_high)

        return vectors_u, vectors_v

    def _compute_eigenvectors(
            self,
            eig_vectors_low: torch.Tensor, eig_vectors_high: torch.Tensor,
            lvl_down: torch.Tensor, lvl_up: torch.Tensor,
            step_B: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        """
        :param eig_vectors_low:
        :param eig_vectors_high:
        :param lvl_down:
        :param lvl_up:
        :param step_B:
        :return: Eigen vectors only states that includes in transitions.
        The shape is [..., num_transitions, K, ], where K is spin system dimension
        The first tensor is low states,
        the second vector is high states
        """
        weights_low, weights_high = self._compute_linear_interpolation_weights(step_B)
        vectors_u, vectors_v = self._compute_eigenvectors_transitions(eig_vectors_low, eig_vectors_high,
                                                                      lvl_down, lvl_up, weights_low, weights_high)
        if self.output_full_eigenvector:
            vector_full_system = self._compute_eigenvector_full_system(eig_vectors_low, eig_vectors_high,
                                                                       lvl_down, lvl_up, weights_low, weights_high)
        else:
            vector_full_system = None

        return (vectors_u, vectors_v), vector_full_system

    def _apply_roots_valid_mask(self, mask_trans: torch.Tensor, mask_trans_i: torch.Tensor,
                               mask_triu: torch.Tensor, step_B: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply transition mask filtering and update related tensors efficiently.
        :param mask_trans: Current transition mask [batch_size, num_transitions]
        :param mask_trans_i: New mask to apply [batch_size, num_transitions]
        :param mask_triu: Upper triangular mask to update [num_transitions]
        :param step_B: Magnetic field steps to filter [batch_size, num_transitions]
        :return:
        tuple: (filtered_mask_trans, updated_mask_triu, filtered_step_B)
        """
        return mask_trans, mask_triu, step_B

    def _split_batch(self, mask_trans: torch.Tensor,
                     mask_triu: torch.Tensor,
                     row_indexes: torch.Tensor):
        """
        :param mask_trans: The mask of True and False of real transitions. The shape is [..., n], where n is valid columns number
        :param mask_triu: The mask of True and False. The shape is [N], where N is number of all transitions. The number of True == n
        :param row_indexes: The indexes where batch was computed. The shape is [...].
        It shows what transitions among global transitions are considered. indexes[indexes==True].shape = [...]
        :return: tuple[mask_triu_updated, row_indexes]

        This musk split the data into sub-batches that for each sub-batch all transitions occurse.
        For example:
            mask_trans = torch.tensor([
            [True, False, False],
            [True, False, False],

            [True, True, True],
            [True, True, True],

            [True, False, True],
            [True, False, True],

            [False, False, True],
            ])

            and mask_triu = torch.Tensor([False, False, True, True, True, False])
            and batch.shape = [7, 3, ...]

        Then in the output the data must be splitted into 4 parts with repect to:
            [True, False, False],
            [True, False, False], with corresponding mask_triu_updated = torch.Tensor([False, False, True, False, False, False])
                                  and batch_updated.shape = [2, 1, ...]

            [True, True, True],
            [True, True, True], with corresponding mask_triu_updated = torch.Tensor([False, False, True, True, True, False])
                                and batch_updated.shape = [2, 3, ...]

            [True, False, True],
            [True, False, True], with corresponding mask_triu_updated = torch.Tensor([False, False, True, False, True, False])
                                 and batch_updated.shape = [2, 2, ...]


            [Flase, False, True], with corresponding mask_triu_updated = torch.Tensor([False, False, False, False, True, False])
                                 and batch_updated.shape = [1, 1, ...]

        """
        original_shape = mask_trans.shape
        flattened_mask = mask_trans.view(-1, original_shape[-1])
        n = original_shape[-1]

        if n <= 63:
            ar = torch.arange(n, dtype=torch.long, device=flattened_mask.device)
            pattern_int = (flattened_mask.long() << ar).sum(dim=-1)
            unique_ints, inverse_indices, counts = torch.unique(pattern_int, return_inverse=True, return_counts=True)
            use_bitpack = True
        else:
            mask_patterns = flattened_mask.int()
            unique_patterns, inverse_indices, counts = torch.unique(mask_patterns, dim=0, return_inverse=True,
                                                                    return_counts=True)
            use_bitpack = False

        # Pre-compute mask_triu true positions for efficiency
        triu_true_positions = torch.where(mask_triu)[0]
        #indexes_true_positions = torch.where(indexes)[0]

        num_unique = len(counts)
        result = []
        for pattern_idx in range(num_unique):
            if use_bitpack:
                unique_int = unique_ints[pattern_idx]
                if unique_int == 0:
                    continue
                bit_mask = ((unique_int >> ar) & 1).bool()
                pattern_true_indices = torch.where(bit_mask)[0]
            else:
                current_pattern = unique_patterns[pattern_idx].bool()
                if not current_pattern.any():
                    continue
                pattern_true_indices = torch.where(current_pattern)[0]

            # Get local indices without sorting
            pattern_local_indices = torch.where(inverse_indices == pattern_idx)[0]
            if len(pattern_local_indices) == 0:
                continue

            # Map to global indices
            row_global_pattern_indices = row_indexes.index_select(0, pattern_local_indices)

            # Create updated mask_triu efficiently
            mask_triu_updated = torch.zeros_like(mask_triu)
            if len(pattern_true_indices) > 0:
                pos_to_set = triu_true_positions[pattern_true_indices]
                mask_triu_updated[pos_to_set] = True
            result.append((mask_triu_updated, row_global_pattern_indices, pattern_local_indices, pattern_true_indices))
            # Select batches using integer indices to avoid bool masking

            #batches_selected = \
            #    [batch[pattern_local_indices].index_select(dim=len(original_shape) - 1, index=pattern_true_indices) for batch in batches_triu] + \
            #    [batch[pattern_local_indices] for batch in batches_general]

        return result

    def _get_resonance_data(self, resonance_field_data, mask_triu, mask_trans, row_indexes):
        result = [
            (mask_triu_new, row_indexes_new, pattern_local_indices, pattern_true_indices, step_B)
            for step_B, mask_trans_i in resonance_field_data
            for mask_triu_new, row_indexes_new, pattern_local_indices, pattern_true_indices in (
                lambda mask_trans_updated, mask_triu_updated, step_B_updated:
                self._split_batch(mask_trans_updated, mask_triu_updated, row_indexes)
            )(*self._apply_roots_valid_mask(mask_trans, mask_trans_i, mask_triu, step_B))
        ]
        return result

    def _iterate_batch(self,
                       batch: tuple[
                                tuple[torch.Tensor, torch.Tensor],
                                tuple[torch.Tensor, torch.Tensor],
                                tuple[torch.Tensor, torch.Tensor],
                                tuple[torch.Tensor, torch.Tensor],
                                torch.Tensor],
                       resonance_frequency: torch.Tensor,
                       baseline_sign: tp.Optional[torch.Tensor], derivative_max: tp.Optional[torch.Tensor]) ->\
            list[tuple[
                    tuple[torch.Tensor, torch.Tensor],
                    tuple[torch.Tensor, torch.Tensor],
                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        """
        :param batch: Tuple with the parameters of the interval:
             (eig_values_low, eig_values_high) - eigen values of Hamiltonian
             at the low and high magnetic fields
             (eig_vectors_low, eig_vectors_high) - eigen vectors of Hamiltonian
             at the low and high magnetic fields
             (energy_derivatives_low, energy_derivatives_high) - the derivatives of the energy at
             low and high magnetic field
             (energy_derivatives_low, energy_derivatives_high) - the derivatives of the energy
             at low and high magnetic field
             (magnetic field low, magnetic field high) - the low and high magnetic fields
             indexes, where mask is valid
        :param resonance_frequency: the resonance frequency
        args are additional arguments to compute resonance mask
        :return:
        """
        eig_values_low, eig_values_high, eig_vectors_low, eig_vectors_high, deriv_low, deriv_high, \
            B_low, B_high, row_indexes, delta_B, lvl_down, lvl_up = self._prepare_inputs(batch)


        diff_eig_low, diff_eig_high, diff_deriv_low, diff_deriv_high =\
            self._compute_raw_differences(eig_values_low, eig_values_high, deriv_low, deriv_high, lvl_down, lvl_up)

        mask_res = self.get_resonance_mask(diff_eig_low, diff_eig_high, B_low, B_high,
                                           resonance_frequency, row_indexes, baseline_sign, derivative_max)

        mask_triu, mask_trans = self._split_mask(mask_res)

        diff_eig_low = diff_eig_low[mask_res]
        diff_eig_high = diff_eig_high[mask_res]

        diff_deriv_low = (delta_B * diff_deriv_low).squeeze(-2)[mask_res]
        diff_deriv_high = (delta_B * diff_deriv_high).squeeze(-2)[mask_res]

        resonance_field_data = self._compute_resonance_fields(diff_eig_low, diff_eig_high, diff_deriv_low,
                                    diff_deriv_high, mask_trans, resonance_frequency)

        resonance_field_data = self._get_resonance_data(resonance_field_data, mask_triu, mask_trans, row_indexes)

        original_shape = mask_trans.shape
        outputs = []
        for mask_triu_new, row_indexes_new, pattern_local_indices, pattern_true_indices, step_B in resonance_field_data:
            resonance_energies = self._compute_resonance_energies(step_B,
                                                                  eig_values_low, eig_values_high,
                                                                  delta_B * deriv_low, delta_B * deriv_high
                                                                  )
            Bres = B_low.squeeze(dim=-1) + step_B * delta_B.squeeze(dim=-1)

            row_idx = pattern_local_indices.view(-1, 1)
            col_idx = pattern_true_indices.view(1, -1)
            idx = [slice(None)] * len(original_shape)
            idx[0] = row_idx
            idx[-1] = col_idx

            resonance_energies_new = resonance_energies[tuple(idx)]
            step_B_new = step_B[tuple(idx)]
            Bres = Bres[tuple(idx)]

            eig_vectors_low_new = eig_vectors_low[pattern_local_indices]
            eig_vectors_high_new = eig_vectors_high[pattern_local_indices]

            valid_lvl_down = lvl_down[mask_triu_new]
            valid_lvl_up = lvl_up[mask_triu_new]

            (vectors_u, vectors_v), vector_full_system = self._compute_eigenvectors(
                eig_vectors_low_new, eig_vectors_high_new, valid_lvl_down, valid_lvl_up, step_B_new)

            out_res = (
                (vectors_u, vectors_v),
                (valid_lvl_down, valid_lvl_up),
                Bres, mask_triu_new,
                row_indexes_new,
                resonance_energies_new,
                vector_full_system
            )
            outputs.append(out_res)
        return outputs

    def forward(self, final_batches, resonance_frequency,
                baselign_sign:tp.Optional[torch.Tensor] = None,
                derivative_max: tp.Optional[torch.Tensor] = None
                ):
        return list(chain.from_iterable(
            self._iterate_batch(batch, resonance_frequency, baselign_sign, derivative_max)
            for batch in final_batches
        ))


class GeneralResonanceLocator(BaseResonanceLocator):
    def _find_three_roots(self, a: torch.Tensor, p: torch.Tensor, q: torch.Tensor):
        """
        coefficients of 3-order polynomial
        """
        p_m3 = torch.sqrt(-p / 3)

        acos_argument = (-q / 2) / (p_m3 ** 3)
        acos_argument = torch.clamp(acos_argument, -1.0, 1.0)

        phi = torch.acos(acos_argument)
        t1 = 2 * p_m3 * torch.cos(phi / 3) - a / 3
        t2 = 2 * p_m3 * torch.cos((phi + 2 * torch.pi) / 3) - a / 3
        t3 = 2 * p_m3 * torch.cos((phi + 4 * torch.pi) / 3) - a / 3

        roots_case1 = torch.stack([t1, t2, t3], dim=-1)  # (..., 3)
        return roots_case1

    def _filter_three_roots(self, three_roots: torch.Tensor) -> torch.Tensor:
        """
        Filter three toorch to make number of inf less.
        :param three_roots: The tensor with the shape [N, 3]. The root can be number or inf
        :return: three_roots with minima infs
        """
        inf_mask = torch.isinf(three_roots)
        perm = inf_mask.int().argsort(dim=-1)
        three_roots = torch.gather(three_roots, dim=-1, index=perm)
        mask = ~torch.all(torch.isinf(three_roots), dim=0)
        three_roots = three_roots[:, mask]
        return three_roots

    def _compute_resonance_fields(self, diff_eig_low, diff_eig_high, diff_deriv_low,
                                  diff_deriv_high, mask_trans, resonance_frequency):
        results = self._find_resonance_roots(diff_eig_low, diff_eig_high,
                                                  diff_deriv_low, diff_deriv_high,
                                                  resonance_frequency)
        if not results or all(roots.numel() == 0 for roots, _ in results):
            return []
        outs = []
        for roots, mask_roots in results:
            pair_mask = torch.clone(mask_trans)
            step_B = torch.zeros_like(mask_trans, dtype=resonance_frequency.dtype)
            pair_mask[mask_trans] = mask_roots
            step_B[pair_mask] = roots
            outs.append((step_B, pair_mask))
        return outs

    def _prepare_roots_mask_batch(self, three_roots: torch.Tensor, one_root: torch.Tensor,
                                  mask_three_roots: torch.Tensor, mask_one_root: torch.Tensor) ->\
            list[tuple[torch.Tensor, torch.Tensor]]:
        """
        :param three_roots: The tensor with the shape [N, 3]. The root can be number or inf
        :param one_root: The tensor with the shape [M, 1].
        :param mask_three_roots: The mask where can be three roots. The shape [N + M]
        :param mask_one_root: ~mask_three_roots: The mask where can be only one root. The shape [N + M]
        :return: the list of the next pairs:
        (roots, mask), where mask has shape [N + M], roots has shape [K], where K is number of True in the mask
        """
        three_roots = self._filter_three_roots(three_roots)
        lines = mask_three_roots.shape[0]
        columns = three_roots.shape[-1]

        out_columns = columns if columns > 0 else 1
        full_roots = torch.full(
            (lines, out_columns),
            float('nan'),
            dtype=three_roots.dtype,
            device=three_roots.device
        )
        if columns > 0:
            full_roots[mask_three_roots] = three_roots

        full_roots[mask_one_root, 0] = one_root.squeeze(-1)
        valid = full_roots.ge(0.0) & full_roots.le(1.0)
        #valid = (full_roots >= 0.0) & (full_roots <= 1.0)
        result = []
        for i in range(valid.shape[-1]):
            mask = valid[:, i]  # shape [B]
            roots = full_roots[mask, i]  # 1D tensor of valid roots in slot i
            result.append((roots, mask))
        return result

    def _find_resonance_roots(self, diff_eig_low: torch.Tensor,
                                     diff_eig_high: torch.Tensor,
                                     diff_deriv_low: torch.Tensor,
                                     diff_deriv_high: torch.Tensor,
                                     resonance_frequency: torch.Tensor):
        """
        Finds the magnetic field value (B_mid) where a resonance occurs by solving a cubic equation
        using the Newton–Raphson method. It suggested that the resonance point is just one

        The cubic polynomial is defined by:
            p3 * t^3 + p2 * t^2 + p1 * t + p0 = 0
        with coefficients constructed from the input parameters:
            p3 = 2 * diff_eig_low - 2 * diff_eig_high + diff_deriv_low + diff_deriv_high
            p2 = -3 * diff_eig_low + 3 * diff_eig_high - 2 * diff_deriv_low - diff_deriv_high
            p1 = diff_deriv_low
            p0 = diff_eig_low - target_resonance

        :param diff_eig_low: Difference of eigenvalues at B_min for the pair, shape compatible with u and v.
        :param diff_eig_high: Difference of eigenvalues at B_max for the pair.
        :param diff_deriv_low: Difference of derivatives at B_min for the pair.
        :param diff_deriv_high: Difference of derivatives at B_max for the pair.
        :param resonance_frequency: The resonance frequency (or energy) to be reached.
        :return: list of tuples (magnetic field, mask)
        Estimated magnetic field values where resonance occurs, shape matching input pair dimensions.
        and mask where transition occurs

        """
        eps = torch.tensor(1e-11, device=diff_eig_high.device)
        tresshold = torch.tensor(1e-10, device=diff_eig_high.device)
        coef_3, coef_2, coef_1, coef_0 = self._compute_cubic_polinomial_coeffs(
            diff_eig_low, diff_eig_high, diff_deriv_low, diff_deriv_high)
        coef_0 = coef_0 - resonance_frequency

        torch.where(coef_3.abs() >= tresshold, coef_3, eps, out=coef_3)
        a = coef_2 / coef_3
        b = coef_1 / coef_3
        c = coef_0 / coef_3

        p = b - a ** 2 / 3
        q = (2 * a ** 3) / 27 - (a * b) / 3 + c

        discriminant = (q / 2) ** 2 + (p / 3) ** 3

        mask_three_roots = discriminant <= 0
        mask_one_root = ~mask_three_roots
        roots_case1 = torch.empty((mask_three_roots.sum(), 3), dtype=a.dtype, device=a.device).fill_(float('inf'))
        roots_case2 = torch.empty((mask_one_root.sum(), 1), dtype=a.dtype, device=a.device).fill_(float('inf'))

        if mask_three_roots.any():
            roots_case1 = self._find_three_roots(a[mask_three_roots], p[mask_three_roots], q[mask_three_roots])
            roots_in_interval = roots_case1.ge(0.0) & roots_case1.le(1.0)
            #roots_in_interval = (roots_case1 >= 0.0) & (roots_case1 <= 1.0)
            roots_case1 = torch.where(roots_in_interval, roots_case1, torch.full_like(roots_case1, float('inf')))

        if mask_one_root.any():
            roots_case2 = self._find_one_root_newton(coef_3[mask_one_root],
                                                     coef_2[mask_one_root],
                                                     coef_1[mask_one_root],
                                                     coef_0[mask_one_root]).unsqueeze(-1)

        result = self._prepare_roots_mask_batch(roots_case1, roots_case2, mask_three_roots, mask_one_root)
        return result

    def _has_rapid_variation(self, res_low: torch.Tensor, res_high: torch.Tensor,
                            B_low: torch.Tensor, B_high: torch.Tensor, deriv_max: torch.Tensor) -> torch.Tensor:
        """
        calculate the criteria that delta_1N < resonance_frequency
        :param res_low: resonance function for the lower magnetic field in the interval. The shape is [..., num_pairs],
        where K is spin system dimension
        :param res_high: resonance function for the higher magnetic field in the interval. The shape is [..., num_pairs],
        where K is spin system dimension
        :param deriv_max: It is a maxima of energy derevative En'(infinity).
        The calculations can be found in original article. The shape is [...]
        :param B_low: It is minima magnetic field of the interval. The shape is [..., 1, 1]
        :param B_high: It is maxima magnetic field of the interval. The shape is [..., 1, 1]
        :return: mask with the shape [...]. If the value is True, the segment could be bisected further.
        """
        threshold = (deriv_max * (B_high - B_low).squeeze()).unsqueeze(-1)
        mask = ((res_low + res_high) / 2).abs() <= threshold
        return mask

    def get_resonance_mask(self, diff_eig_low: torch.Tensor,
                           diff_eig_high: torch.Tensor,
                           B_low: torch.Tensor, B_high: torch.Tensor,
                           resonance_frequency: torch.Tensor, row_indexes: torch.Tensor,
                           baseline_sign_mask: torch.Tensor, deriv_max: torch.Tensor):
        """
        :param diff_eig_low:
        :param diff_eig_high:
        :param resonance_frequency:
        :return:
        """
        res_high = diff_eig_high - resonance_frequency
        res_low = diff_eig_low - resonance_frequency
        mask_sign_change = (res_low * res_high <= 0)
        mask_delta = self._has_rapid_variation(res_low, res_high, B_low, B_high, deriv_max.index_select(0, row_indexes))
        return torch.where(baseline_sign_mask.index_select(0, row_indexes).unsqueeze(-1), mask_delta, mask_sign_change)

    def _apply_roots_valid_mask(self, mask_trans: torch.Tensor, mask_trans_i: torch.Tensor,
                               mask_triu: torch.Tensor, step_B: torch.Tensor) ->\
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply transition mask filtering and update related tensors efficiently.
        :param mask_trans: Boolean current transition mask [batch_size, num of old valid transitions]
        :param mask_trans_i: Boolean New mask to apply [batch_size, num of new valid transitions]
        :param mask_triu: Boolean upper triangular mask to update [num of all possible transitions]
        :param step_B: Magnetic field steps to filter [batch_size, num of old valid transitions]
        :return:
        tuple: (filtered_mask_trans, updated_mask_triu, filtered_step_B)
        """
        mask_triu_updated = mask_triu.clone()
        combined_mask = mask_trans & mask_trans_i
        active_transitions = combined_mask.any(dim=-2)
        filtered_mask_trans = combined_mask[..., active_transitions]
        filtered_step_B = step_B[..., active_transitions]
        mask_triu_updated[mask_triu] = active_transitions

        return filtered_mask_trans, mask_triu_updated, filtered_step_B


class ResField(nn.Module):
    def __init__(self, spin_system_dim: int,
                 mesh_size: torch.Size,
                 batch_dims: torch.Size | tuple,
                 eigen_finder: BaseEigenSolver = EighEigenSolver(), output_full_eigenvector: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param eigen_finder: The eigen solver that should find eigen values and eigen vectors
        """
        super().__init__()
        self.register_buffer('spin_system_dim', torch.tensor(spin_system_dim))
        self.output_full_eigenvector = output_full_eigenvector
        self.general_solver = GeneralResonanceIntervalSolver(self.spin_system_dim, eigen_finder=eigen_finder,
                                                             device=device, dtype=dtype)
        self.general_locator = GeneralResonanceLocator(output_full_eigenvector=self.output_full_eigenvector,
                                                       device=device, dtype=dtype)
        self.zerofree_solver = ZeroFreeResonanceIntervalSolver(self.spin_system_dim, eigen_finder=eigen_finder,
                                                               device=device, dtype=dtype)
        self.zerofree_locator = BaseResonanceLocator(output_full_eigenvector=self.output_full_eigenvector,
                                                     device=device, dtype=dtype)
        self.mesh_size = mesh_size
        self.batch_dims = batch_dims
        self.device = device
        self.eigen_finder = eigen_finder

    def _solver_fabric(self, system, F: torch.Tensor, resonance_frequency: torch.Tensor) \
            -> tuple[BaseResonanceIntervalSolver, BaseResonanceLocator, tuple[tp.Any]]:
        """
        :param F: The part of Hamiltonian that doesn't depend on the magnetic field
        :param resonance_frequency: The frequency of resonance
        :return:
        """
        baselign_sign = self._compute_zero_field_resonance(F / resonance_frequency,
                                                           resonance_frequency / resonance_frequency)
        if baselign_sign.all():
            locator = self.general_locator
            interval_solver = self.general_solver
            args = (baselign_sign.flatten(0, -1), system.calculate_derivative_max().flatten(0, -1) / resonance_frequency)
        elif baselign_sign.any():
            locator = self.general_locator
            interval_solver = self.general_solver
            args = (baselign_sign.flatten(0, -1), system.calculate_derivative_max().flatten(0, -1) / resonance_frequency)
        else:
            locator = self.zerofree_locator
            interval_solver = self.zerofree_solver
            args = (None, None)

        return interval_solver, locator, args

    @staticmethod
    def _compute_zero_field_resonance(F: torch.tensor, resonance_frequency: torch.tensor):
        """
        :param F: Magnetic filed free stationary Hamiltonian matrix. The shape is [..., K, K],
        where K is spin system dimension
        :param resonance_frequency: the resonance frequency. The shape is []
        :return: The mask, where True if resonance function > 0, and False otherwise
        """
        eig_values = torch.linalg.eigvalsh(F)
        res_1N = eig_values[..., -1] - eig_values[..., 0] - resonance_frequency
        return res_1N > 0

    def _assign_global_indexes(self,
                               occurrences: list[tuple[int, torch.Tensor, torch.Tensor]]
                               ) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]], int]:
        """
        'occurrences' is list of the next data:
        the batch index
        The row-batch indexes
        The column indexes with respect to the triangular matrix of transitions
        The keep_local. - The pairs that were saved in filter by max
        The algorithm create global indexes

        Example 1
        row = [1 ,2, 3], col = [1, 2]
        row = [1, 2, 3], col = [3, 4]
        the output
        row = [1 , 2, 3], col = [1, 2], glob_idx = [0, 1]
        row = [1 , 2, 3], col = [3,  4], glob_idx = [2, 3]

        Example 2
        row = [1 ,2, 3], col = [1, 2]
        row = [4, 5, 6], col = [1, 2]
        the output
        row = [1 , 2, 3], col = [1, 2], glob_idx = [0, 1]
        row = [4 , 5, 6], col = [1,  2], glob_idx = [0, 1]

        Example 3
        row = [1 ,2], col = [1, 2]
        row = [2, 3], col = [2, 3]
        the output
        row = [1 , 2], col = [1, 2], glob_idx = [0, 1]
        row = [2 , 3], col = [2,  3], glob_idx = [2, 3]

        Example 4
        row = [1 ,2 , 3], col = [1, 2]
        row = [4, 5, 6], col = [2, 3]
        the output
        row = [1 , 2, 3], col = [1, 2], glob_idx = [0, 1]
        row = [4 , 5, 6], col = [2,  3], glob_idx = [1, 2]
        """
        col_to_slots = {}

        out: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        max_global = -1

        for (batch_idx, rows, cols) in occurrences:
            rows_set = set(rows.tolist())
            assigned_indices: list[int] = []
            for c in cols.tolist():
                slots = col_to_slots.get(int(c), [])
                reused = False
                for slot_idx, (gidx, used_rows) in enumerate(slots):
                    if used_rows.isdisjoint(rows_set):
                        assigned_indices.append(gidx)
                        slots[slot_idx] = (gidx, used_rows.union(rows_set))
                        reused = True
                        break

                if not reused:
                    max_global += 1
                    gidx = max_global
                    assigned_indices.append(gidx)
                    slots.append((gidx, set(rows_set)))
                    col_to_slots[int(c)] = slots

            global_idx_tensor = torch.tensor(assigned_indices, dtype=torch.long)
            out.append((batch_idx, rows, cols, global_idx_tensor))
        return out, max_global + 1

    def _first_pass(self, batches):
        cached_batches = []
        occurences = []
        for bi, batch in enumerate(batches):
            (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up), \
                Bres, mask_triu, row_indexes, resonance_energies, vector_full_system = batch

            col_idx = torch.nonzero(mask_triu, as_tuple=False).squeeze(-1)

            if row_indexes.numel() > 0 and col_idx.numel() > 0:
                cached_batches.append((
                    (vectors_u, vectors_v),
                    (valid_lvl_down, valid_lvl_up),
                    Bres, row_indexes, resonance_energies, vector_full_system
                ))
                occurences.append((bi, row_indexes, col_idx))
        return cached_batches, occurences

    def _combine_resonance_data(self,
                                dtype: torch.dtype,
                                device: torch.device,
                                batches: list[
                               tuple[
                                    tuple[torch.Tensor, torch.Tensor],
                                    tuple[torch.Tensor, torch.Tensor],
                                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None,
                               ]], resonance_frequency: torch.Tensor) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        """
        :param batches: list of next data:
        - tuple of eigen vectors for resonance energy levels of low and high levels
        - tuple of valid indexes of levels between which transition occurs
        - magnetic field of transitions
        - mask_trans - Boolean transition mask [batch_size, num valid transitions]. It is True if transition occurs
        - mask_triu - Boolean triangular mask [total number of all possible transitions].
        It is True if there is at least one element in batch for which transition between energy levels occurs.
        num valid transitions = sum(mask_triu)
        - indexes: Boolean mask of correct elements in batch
        - resonance energies
        - vector_full_system | None. The eigen vectors for all energy levels
        :return:
        torch.Tensor, torch.Tensor
        """

        config_dims = (*self.batch_dims, *self.mesh_size)
        total_batch_size = torch.prod(torch.tensor(config_dims)).item()

        batches, occurrences = self._first_pass(batches)
        occurrences, max_columns = self._assign_global_indexes(occurrences)

        if dtype is torch.float32:
            complex_dtype = torch.complex64
        else:
            complex_dtype = torch.complex128

        vectors_u = torch.zeros((total_batch_size, max_columns, self.spin_system_dim), dtype=complex_dtype,
                                device=device)
        vectors_v = torch.zeros((total_batch_size, max_columns, self.spin_system_dim), dtype=complex_dtype,
                                device=device)
        resonance_energies = torch.zeros((total_batch_size, max_columns, self.spin_system_dim), dtype=dtype,
                                         device=device)

        valid_lvl_down = torch.zeros(max_columns, dtype=torch.long, device=device)
        valid_lvl_up = torch.zeros(max_columns, dtype=torch.long, device=device)
        res_fields = torch.zeros((total_batch_size, max_columns), dtype=dtype, device=device)

        if self.output_full_eigenvector:
            full_eigen_vectors = torch.zeros((total_batch_size, max_columns,
                                              self.spin_system_dim, self.spin_system_dim),
                                              dtype=complex_dtype, device=device)
        else:
            full_eigen_vectors = None

        for bi, rows_batch_idx, col_batch_idx, col_base_idx in occurrences:
            (vectors_u_batch, vectors_v_batch), (valid_lvl_down_batch, valid_lvl_up_batch),\
                resonance_field_batch, row_indexes, resonance_energies_batch, eigen_vectors_batched = batches[bi]

            if row_indexes.numel() > 0 and col_base_idx.numel() > 0:
                vectors_u[row_indexes[:, None], col_base_idx, :] = vectors_u_batch
                vectors_v[row_indexes[:, None], col_base_idx, :] = vectors_v_batch

                valid_lvl_down[col_base_idx] = valid_lvl_down_batch
                valid_lvl_up[col_base_idx] = valid_lvl_up_batch

                resonance_energies[row_indexes[:, None], col_base_idx, :] = resonance_energies_batch

                res_fields[row_indexes[:, None], col_base_idx] = resonance_field_batch
                if self.output_full_eigenvector:
                    full_eigen_vectors[row_indexes[:, None], col_base_idx, :, :] = eigen_vectors_batched

        vectors_u = vectors_u.view(*config_dims, max_columns, self.spin_system_dim)
        vectors_v = vectors_v.view(*config_dims, max_columns, self.spin_system_dim)
        resonance_energies = resonance_energies.view(*config_dims, max_columns, self.spin_system_dim)
        res_fields = res_fields.view(*config_dims, max_columns)

        if self.output_full_eigenvector:
            full_eigen_vectors = full_eigen_vectors.view(*config_dims, max_columns,
                                                         self.spin_system_dim, self.spin_system_dim)

        return (vectors_u, vectors_v), (valid_lvl_down, valid_lvl_up),\
            res_fields, resonance_energies * resonance_frequency, full_eigen_vectors

    def forward(self, sample: spin_system.BaseSample,
                 resonance_frequency: torch.Tensor,
                 B_low: torch.Tensor, B_high: torch.Tensor, F: torch.Tensor, Gz: torch.Tensor) ->\
            tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        :param sample: The sample for which the resonance parameters need to be found
        :param resonance_frequency: the resonance frequency. The shape is []
        :param B_low: low limit of magnetic field intervals. The shape is [batch_dim]
        :param B_high: high limit of magnetic field intervals. The shape is [batch_dim]
        :param F: The magnetic free part of Hamiltonian
        :param Gz: z-part of Zeeman magnetic field term B * Gz
        :return: list of next data:
        - tuple of the eigen vectors of high transition states and of low transition states and Vi and Vj where i>j
        is EPR transition
        - tuple of valid indexes of levels between which transition occurs
        - magnetic field of transitions
        - resonance energies
        - vector_full_system | None. The eigen vectors for all energy levels
        """
        interval_solver, locator, args = self._solver_fabric(sample, F, resonance_frequency)

        batches = interval_solver(
            F.flatten(0, -3) / resonance_frequency,
            Gz.flatten(0, -3) / resonance_frequency, B_low.flatten(0, -1),
            B_high.flatten(0, -1), resonance_frequency / resonance_frequency, *args)

        batches = locator(batches, resonance_frequency / resonance_frequency, *args)
        out = self._combine_resonance_data(dtype=resonance_frequency.dtype,
                                           device=Gz.device, batches=batches, resonance_frequency=resonance_frequency)
        return out







