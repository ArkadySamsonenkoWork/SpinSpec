import warnings
from abc import ABC, abstractmethod
import typing as tp

import torch
from torch import nn

class BaseEigenSolver(ABC):
    @abstractmethod
    def __call__(self, F: torch.Tensor, G: torch.Tensor, B: torch.Tensor):
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
    def __call__(self, F: torch.Tensor, G: torch.Tensor, B: torch.Tensor):
        return torch.linalg.eigh(F + G * B)

    def compute_eigenvalues(self, F: torch.Tensor, G: torch.Tensor, B: torch.Tensor):
        return torch.linalg.eigvalsh(F + G * B)


def has_sign_change(res_low: torch.Tensor, res_high: torch.Tensor) -> torch.Tensor:
    """
    calculate the criteria that delta_1N < resonance_frequency
    :param res_low: resonance function for the lower magnetic field in the interval. The shape is [..., K, K],
    where K is spin system dimension
    :param res_high: resonance function for the higher magnetic field in the interval. The shape is [..., K, K],
    where K is spin system dimension
    :return: mask with the shape [...]. If the value is True, the resonance function changes
    sign and segment can ve bisected further
    """
    mask = ((res_low * res_high) <= 0).any(dim=(-1, -2))
    return mask


def has_rapid_variation(res_low: torch.Tensor, res_high: torch.Tensor,
                            deriv_max: torch.Tensor, B_low: torch.Tensor, B_high: torch.Tensor) -> torch.Tensor:
    """
    calculate the criteria that delta_1N < resonance_frequency
    :param res_low: resonance function for the lower magnetic field in the interval. The shape is [..., K, K],
    where K is spin system dimension
    :param res_high: resonance function for the higher magnetic field in the interval. The shape is [..., K, K],
    where K is spin system dimension
    :param deriv_max: It is a maxima of energy derevative En'(infinity).
    The calculations can be found in original article. The shape is [...]
    :param B_low: It is minima magnetic field of the interval. The shape is [..., 1, 1]
    :param B_high: It is maxima magnetic field of the interval. The shape is [..., 1, 1]
    :return: mask with the shape [...]. If the value is True, the segment could be bisected further.
    """
    mask = (((res_low + res_high) / 2).abs() <= deriv_max * (B_high - B_low)).any(dim=(-1, -2))
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
class BaseResonanceIntervalSolver(ABC):
    """
    Base class for algorithm of resonance interval search
    """
    def __init__(self, eigen_finder: tp.Optional[BaseEigenSolver] = EighEigenSolver(), r_tol: float = 1e-5,
        max_iterations: float=100):
        self.eigen_finder = eigen_finder
        self.r_tol = torch.tensor(r_tol)
        self.max_iterations = torch.tensor(max_iterations)

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

        K = eig_values_low.shape[-1]  # Number of states
        u, v = torch.triu_indices(K, K, offset=1, device=eig_values_low.device)

        res_low = \
            (eig_values_low.unsqueeze(-2)[..., v] - eig_values_low.unsqueeze(-2)[..., u]).squeeze(
                -2) - resonance_frequency
        res_high = \
            (eig_values_high.unsqueeze(-2)[..., v] - eig_values_high.unsqueeze(-2)[..., u]).squeeze(
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
                        B_low: torch.Tensor, B_high: torch.Tensor, resonance_frequency: torch.Tensor, *args, **kwargs):
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
                                                       resonance_frequency, *args, **kwargs)

        return torch.logical_and(mask_monotonically, mask_loop_dependant)

    @abstractmethod
    def loop_dependant_mask(self, *args, **kwargs):
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
        return torch.einsum('...bi,...ij,...bj->...b', torch.conj(eigen_vector), G, eigen_vector).real

    def compute_error(self, eig_values_low: torch.Tensor, eig_values_mid: torch.Tensor,
                      eig_values_high: torch.Tensor,
                      eig_vectors_low: torch.Tensor,
                      eig_vectors_high: torch.Tensor,
                      B_low: torch.Tensor, B_high: torch.Tensor,
                      G: torch.Tensor,
                      indexes: torch.Tensor):
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
        :param indexes: Indexes where Gz must be sliced. The bool tensor with the shape of the initial shape [...]
        :param G: The magnetic field dependant part of the Hamiltonian: F + G * B. The shape is [..., K, K]
        :return: epsilon is epsilon mistake. The tensor with the shape [...]
        """

        derivatives_low = self._compute_derivative(eig_vectors_low, G[indexes])
        derivatives_high = self._compute_derivative(eig_vectors_high, G[indexes])
        eig_values_estimation = 0.5 * (eig_values_high + eig_values_low) +\
                                    (B_high - B_low) / 8 * (derivatives_high - derivatives_low)
        epsilon = 2 * (eig_values_estimation - eig_values_mid).abs().max(dim=-1)[0]
        return epsilon, (derivatives_low, derivatives_high)

    @abstractmethod
    def determine_split_masks(self, *args, **kwargs):
        pass

    def assemble_current_batches(self,
                     eig_values_low, eig_values_mid, eig_values_high,
                     eig_vectors_low, eig_vectors_mid, eig_vectors_high,
                     B_low, B_mid, B_high, indexes,
                     resonance_frequency, *args, **kwargs):
        new_intervals = []

        mask_left, mask_right = self.determine_split_masks(eig_values_low, eig_values_mid, eig_values_high,
                                                    B_low, B_mid, B_high,
                                                    indexes, resonance_frequency, *args, **kwargs)

        mask_and = torch.logical_and(mask_left, mask_right)
        mask_xor = torch.logical_xor(mask_left, mask_right)

        # Process and case. It means that both intervals have resonance

        if mask_and.any():
            indexes_and = indexes.clone()
            indexes_and[indexes_and == True] = mask_and

            new_intervals.append(
                ((eig_values_low[mask_and], eig_values_mid[mask_and]),
                 (eig_vectors_low[mask_and], eig_vectors_mid[mask_and]),
                 (B_low[mask_and], B_mid[mask_and]),
                 indexes_and)
            )
            new_intervals.append(
                ((eig_values_mid[mask_and], eig_values_high[mask_and]),
                 (eig_vectors_mid[mask_and], eig_vectors_high[mask_and]),
                 (B_mid[mask_and], B_high[mask_and]),
                 indexes_and)
            )

        # Process XOR case. It means that only one interval has resonance.
        # Note, that it is impossible that none interval has resonance

        if mask_xor.any():
            new_intervals.append(
                self._compute_xor_interval(
                    mask_xor,
                    mask_left,
                    mask_right,
                    eig_values_low, eig_values_mid, eig_values_high,
                    eig_vectors_low, eig_vectors_mid, eig_vectors_high,
                    B_low, B_mid, B_high,
                    indexes,
                )
            )
        return new_intervals

    def _compute_xor_interval(self,
            mask_xor: torch.Tensor,
            mask_left: torch.Tensor,
            mask_right: torch.Tensor,
            eig_values_low, eig_values_mid, eig_values_high,
            eig_vectors_low, eig_vectors_mid, eig_vectors_high,
            B_low, B_mid, B_high,
            indexes: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        :param mask_xor: the xor mask of the left and right mask
         It means that resonance happens in the one interval left or right. The shape is [...]
        :param mask_left: the left mask with the values in batches where resonance happens. The shape is [...]
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
        :param indexes: Indexes where Gz must be sliced. The bool tensor with the shape of the INITIAL batch-mesh shape
        :return: tuple of eig_values, eig_vectors, magnetic fields, and new indexes
        """
        # Get adjusted intervals for XOR case

        mask_left = mask_left[mask_xor]
        # Select boundaries based on active mask side
        B_low = torch.where(mask_left.unsqueeze(-1).unsqueeze(-1), B_low[mask_xor], B_mid[mask_xor])
        B_high = torch.where(mask_left.unsqueeze(-1).unsqueeze(-1), B_mid[mask_xor], B_high[mask_xor])
        # Select corresponding eigenvalues/vectors
        eig_values_low = torch.where(mask_left.unsqueeze(1), eig_values_low[mask_xor], eig_values_mid[mask_xor])
        eig_values_high = torch.where(mask_left.unsqueeze(1), eig_values_mid[mask_xor], eig_values_high[mask_xor])
        eig_vectors_low = torch.where(mask_left.unsqueeze(-1).unsqueeze(-1), eig_vectors_low[mask_xor], eig_vectors_mid[mask_xor])
        eig_vectors_high = torch.where(mask_left.unsqueeze(-1).unsqueeze(-1), eig_vectors_mid[mask_xor], eig_vectors_high[mask_xor])

        # Update tracking indexes
        indexes = indexes.clone()
        indexes[indexes == True] = mask_xor

        return (
            (eig_values_low, eig_values_high),
            (eig_vectors_low, eig_vectors_high),
            (B_low, B_high),
            indexes
        )

    def _iterate_batch(self, batch: tuple[
                                 tuple[torch.Tensor, torch.Tensor],
                                 tuple[torch.Tensor, torch.Tensor],
                                 tuple[torch.Tensor, torch.Tensor],
                                 torch.Tensor],
                             F: torch.Tensor, Gz: torch.Tensor,
                             resonance_frequency: torch.Tensor, a_tol: torch.Tensor, *args, **kwargs):
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
        (eig_values_low, eig_values_high), (eig_vectors_low, eig_vectors_high), (B_low, B_high), indexes = batch
        B_mid = (B_low + B_high) / 2
        eig_values_mid, eig_vectors_mid = torch.linalg.eigh(F[indexes] + Gz[indexes] * B_mid)
        # It is only one single
        # point where gradient should be calculated
        error, (derivatives_low, derivatives_high) = \
            self.compute_error(eig_values_low, eig_values_mid, eig_values_high,
                               eig_vectors_low, eig_vectors_high,
                               B_low, B_high, Gz, indexes
                               )

        converged_mask = (error <= a_tol).any(dim=-1)
        # На этом шаге нужно также разделить инетервал на две части. eig_values_mid, eig_vectors_mid уже посчитаны!!
        # Но нужно тогда пересчитывать derivatives
        if converged_mask.any():
            indexes_conv = indexes.clone()
            indexes_conv[indexes_conv == True] = converged_mask
            final_batches.append((
                (eig_values_low[converged_mask], eig_values_high[converged_mask]),
                (eig_vectors_low[converged_mask], eig_vectors_high[converged_mask]),
                (derivatives_low[converged_mask], derivatives_high[converged_mask]),
                (B_low[converged_mask], B_high[converged_mask]),
                indexes_conv,
            ))

        active_mask = ~converged_mask
        if not active_mask.any():
            return [], final_batches

        # Update active components.
        B_low = B_low[active_mask]
        B_high = B_high[active_mask]
        B_mid = B_mid[active_mask]

        eig_values_low = eig_values_low[active_mask]
        eig_values_mid = eig_values_mid[active_mask]
        eig_values_high = eig_values_high[active_mask]

        eig_vectors_low = eig_vectors_low[active_mask]
        eig_vectors_mid = eig_vectors_mid[active_mask]
        eig_vectors_high = eig_vectors_high[active_mask]
        # indexes = batch["indexes"]
        indexes[indexes == True] = active_mask
        new_batches = self.assemble_current_batches(
            eig_values_low, eig_values_mid, eig_values_high,
            eig_vectors_low, eig_vectors_mid, eig_vectors_high,
            B_low, B_mid, B_high, indexes, resonance_frequency,
            *args, **kwargs)
        return new_batches, final_batches

    # Вероятно, нужно будет поменять на дикты. Но будут проблемы с jit-компиляцией
    def __call__(self, F: torch.Tensor, Gz: torch.Tensor,
                                B_low: torch.Tensor, B_high: torch.Tensor,
                                resonance_frequency: torch.Tensor, *args, **kwargs) ->\
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
        eig_values, eig_vectors = torch.linalg.eigh(Hamiltonians)
        eig_values_low, eig_values_high = eig_values[..., 0, :], eig_values[..., 1, :]
        eig_vectors_low, eig_vectors_high = eig_vectors[..., 0, :, :], eig_vectors[..., 1, :, :]

        iterations = 0

        # True means that continue divide
        active_mask = self.check_resonance(eig_values_low, eig_values_high,
                                           B_low, B_high, resonance_frequency, *args, **kwargs
                                           )
        if torch.all(~active_mask):
            warnings.warn("There are no resonance in the interval")
        final_batches = []
        current_batches = [(
            (eig_values_low[active_mask], eig_values_high[active_mask]),
            (eig_vectors_low[active_mask], eig_vectors_high[active_mask]),
            (B_low[active_mask], B_high[active_mask]),
            active_mask
        )]

        while current_batches:
            final_batches = []
            iteration_results = [self._iterate_batch(batch, F, Gz, resonance_frequency, a_tol, *args, **kwargs)
                                 for batch in current_batches]
            current_batches = [current_batch for batches in iteration_results for current_batch in batches[0]]
            final_batches.extend([current_batch for batches in iteration_results for current_batch in batches[1]])

            iterations += 1
            if iterations >= self.max_iterations:
                warnings.warn(f"The max iteration number was overbet {self.max_iterations}")

        # locate_resonance_fields(final_batches, resonance_frequency)
        return final_batches


class GeneralResonanceIntervalSolver(BaseResonanceIntervalSolver):
    """
    Find resonance interval for the general Hamiltonian case.
    The general case determines form the condition:
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
        mask_delta = has_rapid_variation(res_low, res_high, deriv_max, B_low, B_high)
        mask_sign_change = has_sign_change(res_low, res_high)
        return torch.where(baseline_sign_mask, mask_delta, mask_sign_change)

    def determine_split_masks(self, eig_values_low, eig_values_mid, eig_values_high,
                                                    B_low, B_mid, B_high,
                                                    indexes, resonance_frequency, baseline_sign, deriv_max):

        mask_left = self.check_resonance(eig_values_low, eig_values_mid, B_low, B_mid,
                                        resonance_frequency, baseline_sign[indexes], deriv_max[indexes])
        mask_right = self.check_resonance(eig_values_mid, eig_values_high, B_mid, B_high,
                                         resonance_frequency, baseline_sign[indexes], deriv_max[indexes])

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
                        resonance_frequency: torch.Tensor
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
                            B_low, B_mid, B_high, indexes, resonance_frequency):
        mask_left = self.check_resonance(eig_values_low, eig_values_mid, B_low, B_mid,
                                         resonance_frequency)
        mask_right = self.check_resonance(eig_values_mid, eig_values_high, B_mid, B_high,
                                          resonance_frequency)
        return mask_left, mask_right



# Может это нужно делать через разреженные матрицы.... Я не понимаю...
# Я считаю все произведения U Gx V даже, если переходов нет. Если переходов нет, то вес ноль. Может,
# это стоит оптимизировать и сделать опреацию дешевле и считать только в valid_u и valid_b.
# Но тогда стоятся лишние струкутуры
# Может быть стоит разделять батчи дальше по u и v....
# Дважды считаю коэффициенты полинома. Нужно будет переделать. Один раз для нахождения корней,
# один раз для получения энергий.
# Возможно, стоит избавиться от двух масок.
class BaseResonanceLocator:
    def __init__(self, max_iterations=10, tolerance=1e-12, accuracy=1e-4):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.accuracy = accuracy

    def _compute_cubic_polinomial_coeffs(self, eig_low, eig_high, deriv_low, deriv_high):
        coef_3 = 2 * eig_low - 2 * eig_high + deriv_low + deriv_high
        coef_2 = -3 * eig_low + 3 * eig_high - 2 * deriv_low - deriv_high
        coef_1 = deriv_low
        coef_0 = eig_low
        return (coef_3, coef_2, coef_1, coef_0)

    def _find_resonance_field_newton(self, diff_eig_low, diff_eig_high, diff_deriv_low, diff_deriv_high,
                                resonance_frequency):
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

        Parameters:
            diff_eig_low (Tensor): Difference of eigenvalues at B_min for the pair, shape compatible with u and v.
            diff_eig_high (Tensor): Difference of eigenvalues at B_max for the pair.
            diff_deriv_low (Tensor): Difference of derivatives at B_min for the pair.
            diff_deriv_high (Tensor): Difference of derivatives at B_max for the pair.
            resonance_frequency (float): The resonance frequency (or energy) to be reached.

        Returns:
            Tensor: Estimated magnetic field values where resonance occurs, shape matching input pair dimensions.
        """
        max_iterations = 50

        (coef_3, coef_2, coef_1, coef_0) = self._compute_cubic_polinomial_coeffs(
            diff_eig_low, diff_eig_high, diff_deriv_low, diff_deriv_high)
        coef_0 -= resonance_frequency
        t = - diff_eig_low / (diff_eig_high - diff_eig_low)
        for _ in range(max_iterations):
            poly_val = coef_3 * t ** 3 + coef_2 * t ** 2 + coef_1 * t + coef_0
            poly_deriv = 3 * coef_3 * t ** 2 + 2 * coef_2 * t + coef_1
            delta = poly_val / (poly_deriv + self.tolerance)
            t -= delta
            if (delta.abs() < self.accuracy).all():
                break
        return t

    def get_resonance_mask(self, diff_eig_low, diff_eig_high, resonance_frequency):
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

    def _compute_resonance_fields(self, diff_eig_low, diff_eig_high, diff_deriv_low,
                                  diff_deriv_high, mask_trans, resonance_frequency):
        step_B = torch.zeros_like(mask_trans, dtype=torch.float32)
        step_B[mask_trans] = self._find_resonance_field_newton(diff_eig_low, diff_eig_high,
                                                  diff_deriv_low, diff_deriv_high,
                                                  resonance_frequency)
        return step_B


    def _compute_resonance_energies(self, step_B, eig_values_low, eig_values_high, deriv_low, deriv_high):
        step_B = step_B.unsqueeze(-1)
        (coef_3, coef_2, coef_1, coef_0) = self._compute_cubic_polinomial_coeffs(
            eig_values_low, eig_values_high, deriv_low, deriv_high)
        energy = coef_3 * step_B ** 3 + coef_2 * step_B ** 2 + coef_1 * step_B + coef_0
        return energy

    def _interpolate_vectors(self, vec_low, vec_high, weights_low, weights_high):
        # Compute inner products along the eigenvector component axis
        inner_product = torch.sum(torch.conj(vec_low) * vec_high, dim=-1, keepdim=True)
        inner_product_abs = torch.clamp(torch.abs(inner_product), min=1e-12)
        phase_factor = inner_product / inner_product_abs
        vec_high_aligned_down = vec_high * torch.conj(phase_factor)
        vectors_u = vec_low * weights_low + vec_high_aligned_down * weights_high
        return vectors_u


    def _iterate_batch(self,
                       batch: tuple[
                                tuple[torch.Tensor, torch.Tensor],
                                tuple[torch.Tensor, torch.Tensor],
                                tuple[torch.Tensor, torch.Tensor],
                                tuple[torch.Tensor, torch.Tensor],
                                torch.Tensor],
                       resonance_frequency: torch.Tensor):
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
             (energy_derivatives_low, energy_derivatives_high) - the derivatives of the energy
             at low and high magnetic field
             indexes, where mask is valid
        :param resonance_frequency: the resonance frequency
        :return:
        """
        (eig_values_low, eig_values_high), (eig_vectors_low, eig_vectors_high),\
            (deriv_low, deriv_high), (B_low, B_high), indexes = batch

        delta_B = B_high - B_low  # [..., 1, 1]
        shape = eig_values_low.shape  # shape is torch.Size([300, 4])
        K = shape[-1]
        lvl_down, lvl_up = torch.triu_indices(K, K, offset=1)
        eig_values_low = eig_values_low.unsqueeze(-2)  # [..., 1, K]
        eig_values_high = eig_values_high.unsqueeze(-2)
        deriv_low = deriv_low.unsqueeze(-2)
        deriv_high = deriv_high.unsqueeze(-2)

        diff_eig_low = (eig_values_low[..., lvl_up] - eig_values_low[..., lvl_down]).squeeze(-2)
        diff_eig_high = (eig_values_high[..., lvl_up] - eig_values_high[..., lvl_down]).squeeze(-2)

        mask = self.get_resonance_mask(diff_eig_low, diff_eig_high, resonance_frequency)
        mask_triu = mask.any(dim=0)  # For some u and v there are no transitions
        mask_trans = mask[..., mask_triu]

        #valid_indices = torch.nonzero(mask, as_tuple=True)[0]  # Get valid indices

        diff_eig_low = diff_eig_low[mask]
        diff_eig_high = diff_eig_high[mask]

        diff_deriv_low = (delta_B * (deriv_low[..., lvl_up] - deriv_low[..., lvl_down])).squeeze(-2)[mask]
        diff_deriv_high = (delta_B * (deriv_high[..., lvl_up] - deriv_high[..., lvl_down])).squeeze(-2)[mask]
        step_B = self._compute_resonance_fields(diff_eig_low, diff_eig_high, diff_deriv_low,
                                    diff_deriv_high, mask_trans, resonance_frequency)
        # Correctly apply mask to u and v
        #u_exp = u.unsqueeze(0).expand(mask.size(0), -1)  # shape: [300, 6]
        #v_exp = v.unsqueeze(0).expand(mask.size(0), -1)  # shape: [300, 6]

        resonance_energies = self._compute_resonance_energies(step_B,
                                                              eig_values_low, eig_values_high,
                                                              delta_B * deriv_low, delta_B * deriv_high)
        valid_lvl_down = lvl_down[mask_triu]
        valid_lvl_up = lvl_up[mask_triu]
        weights_low, weights_high = self._compute_linear_interpolation_weights(step_B)

        # Get the eigenvector columns corresponding to the levels for interpolation
        #vec_low_down = eig_vectors_low[..., valid_lvl_down, :]
        #vec_high_down = eig_vectors_high[..., valid_lvl_down, :]

        vec_low_down = eig_vectors_low[..., :, valid_lvl_down].transpose(-1, -2)
        vec_high_down = eig_vectors_high[..., :, valid_lvl_down].transpose(-1, -2)

        #vec_low_up = eig_vectors_low[..., valid_lvl_up, :]  # shape: [..., num_transitions, N]
        #vec_high_up = eig_vectors_high[..., valid_lvl_up, :]  # same shape

        vec_low_up = eig_vectors_low[..., :, valid_lvl_up].transpose(-1, -2)  # shape: [..., num_transitions, N]
        vec_high_up = eig_vectors_high[..., :, valid_lvl_up].transpose(-1, -2)  # same shape

        vectors_u = self._interpolate_vectors(vec_low_down, vec_high_down, weights_low, weights_high)
        vectors_v = self._interpolate_vectors(vec_low_up, vec_high_up, weights_low, weights_high)
        return (
            (vectors_u, vectors_v),
            (valid_lvl_down, valid_lvl_up),
            B_low.squeeze(dim=-1) + step_B * delta_B.squeeze(dim=-1),
            mask_trans, mask_triu,
            indexes,
            resonance_energies)

    def __call__(self, final_batches, resonance_frequency):
        final_batches = [self._iterate_batch(batch, resonance_frequency) for batch in final_batches]
        return final_batches


class ResField:
    def __init__(self, eigen_finder: BaseEigenSolver = EighEigenSolver()):
        """
        :param eigen_finder: The eigen solver that should find eigen values and eigen vectors
        """
        self.eigen_finder = eigen_finder

    def _solver_fabric(self, F: torch.Tensor, resonance_frequency: torch.Tensor) \
            -> tuple[BaseResonanceIntervalSolver, BaseResonanceLocator, tuple[tp.Any]]:
        """
        :param system:
        :param F: The part of Hamiltonian that doesn't depend on the magnetic field
        :param resonance_frequency: The frequency of resonance
        :return:
        """
        baselign_sign = self._compute_zero_field_resonance(F, resonance_frequency)
        if baselign_sign.all():
            raise NotImplementedError
        elif baselign_sign.any():
            interval_solver = GeneralResonanceIntervalSolver(eigen_finder=self.eigen_finder)
            raise NotImplementedError
        else:
            locator = BaseResonanceLocator()
            interval_solver = ZeroFreeResonanceIntervalSolver(eigen_finder=self.eigen_finder)
            #deriv_max = system.calculate_derivative_max()
            args = ()
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

    def __call__(self, system, resonance_frequency, B_low: torch.Tensor, B_high: torch.Tensor, F, Gz):
        interval_solver, locator, args = self._solver_fabric(F, resonance_frequency)
        bathces = interval_solver(F, Gz, B_low, B_high, resonance_frequency, *args)
        bathces = locator(bathces, resonance_frequency)
        return bathces


