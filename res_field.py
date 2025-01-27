import warnings

import torch
from torch import nn


def compute_resonance_functions(eig_values_low: torch.Tensor, eig_values_high: torch.Tensor,
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

    res_low = eig_values_low[..., :, None] - eig_values_low[..., None, :] - resonance_frequency
    res_high = eig_values_high[..., :, None] - eig_values_high[..., None, :] - resonance_frequency

    #res = eig_values[..., :, None] - eig_values[..., None, :] - resonance_frequency
    #res_low, res_high = res[..., 0, :, :], res[..., 1, :, :]

    return res_low, res_high


def has_monotonically_rule(eig_values_low: torch.Tensor, eig_values_high: torch.Tensor,
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


def has_sign_change(res_low: torch.Tensor, res_high: torch.Tensor) -> torch.Tensor:
    """
    calculate the criteria that delta_1N < resonance_frequency
    :param res_low: resonance function for the lower magnetic field in the interval. The shape is [..., K, K],
    where K is spin system dimension
    :param res_high: resonance function for the higher magnetic field in the interval. The shape is [..., K, K],
    where K is spin system dimension
    :return: mask with the shape [...]. If the value is True, the segment could be bisected further.
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
    :param B_low: It is minima magnetic field of the interval. The shape is [...]
    :param B_high: It is maxima magnetic field of the interval. The shape is [...]
    :return: mask with the shape [...]. If the value is True, the segment could be bisected further.
    """

    #mask = (res_low * res_high).abs() > deriv_max * (B_high - B_low)  # Для всех состояний, то резонанса нет
    #mask = ~torch.all(mask, dim=(-1, -2))
    mask = (res_low * res_high).abs() <= deriv_max * (B_high - B_low)
    mask = torch.any(mask, dim=(-1, -2))
    return mask


def get_zero_resonance_baseline(F: torch.tensor, resonance_frequency: torch.tensor):
    """
    :param F: Magnetic filed free stationary Hamiltonian matrix. The shape is [..., K, K],
    where K is spin system dimension
    :param resonance_frequency: the resonance frequency. The shape is []
    :return: The mask, where True if resonance function > 0, and False otherwise
    """
    eig_values, _ = torch.linalg.eigh(F)
    res_1N = eig_values[..., -1] - eig_values[..., 0] - resonance_frequency
    return res_1N > 0


class Tester(nn.Module):
    def __init__(self, F: torch.Tensor, deriv_max: torch.tensor, resonance_frequency: float):
        super().__init__()
        self.register_buffer('resonance_frequency', torch.tensor(resonance_frequency))
        self.register_buffer("deriv_max", deriv_max)  # [...]
        self.register_buffer("baseline_sign_mask", self._compute_baseline_mask(F))  # [...]

    def _compute_baseline_mask(self, F: torch.Tensor):
        return get_zero_resonance_baseline(F, self.resonance_frequency)

    def forward(self, eig_values_low: torch.Tensor, eig_values_high: torch.Tensor, B_low, B_high):
        """
        Compute the error after division of the interval
        :param eig_values_low: energies in the ascending order at B_low magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_values_high: energies in the ascending order at B_high magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param B_low: It is minima magnetic field of the interval. The shape is [..., L]
        :param B_high: It is maxima magnetic field of the interval. The shape is [..., L]
        :return: mask with the shape [..., L].  If it is true, the interval could be bisected further
        """
        mask_monotonically = has_monotonically_rule(
            eig_values_low, eig_values_high, self.resonance_frequency)  # [..., L]

        res_low, res_high = compute_resonance_functions(
            eig_values_low, eig_values_high, self.resonance_frequency)
        rapid_variation = has_rapid_variation(res_low, res_high, self.deriv_max, B_low, B_high)
        sign_change = has_sign_change(res_low, res_high)

        mask_delta = torch.where(condition=self.baseline_sign_mask, input=rapid_variation, other=sign_change)
        return mask_monotonically * mask_delta


# Must me rebuild to speed up.
def get_resonance_intervals(tester: Tester, F: torch.Tensor, Gz: torch.Tensor, B_low, B_high, resonance_frequency):
    """
    Calculate the resonance intervals, where the resonance field is possible
    :param F: Magnetic filed free stationary Hamiltonian matrix. The shape is [..., K, K],
    where K is spin system dimension
    :param Gz: Magnetic field dependant part of stationary Hamiltonian with the shape [..., K, K].
    :param B_low: The start of the interval to find roots. The shape is [...]
    :param B_high: The end of the interval to find roots. The shape is [...]
    :return:
    """
    r_tol = 1e-4
    max_iterations = 100
    a_tol = resonance_frequency * r_tol
    def compute_error(eig_values_low: torch.Tensor,
                      eig_values_high: torch.Tensor,
                      eig_values_middle: torch.Tensor,
                      eig_vectors_low: torch.Tensor,
                      eig_vectors_high: torch.Tensor,
                      B_low: torch.Tensor, B_high: torch.Tensor):
        """
        Compute the error after division of the interval
        :param eig_values_low: energies in the ascending order at B_low magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_values_high: energies in the ascending order at B_high magnetic field.
        The shape is [..., K], where K is spin system dimension.
        :param eig_values_middle: energies in the ascending order at B_middle magnetic field.
        The shape is [..., K], where K is spin system dimension. B_middle = (B_low + B_high) / 2
        :param eig_vectors_low: eigen vectors corresponding eig_values_low. The shape is [..., K, K],
        where K is spin system dimension
        :param eig_vectors_high: eigen vectors corresponding eig_values_high. The shape is [..., K, K],
        where K is spin system dimension
        :param B_low: The lower magnetic field The shape is [...]
        :param B_high: The higher magnetic field The shape is [...]
        :return: tuple: epsilon is epsilon mistake. The tensor with the shape [...]
                        eig_values_middle are energies in the ascending order in the middle of the interval
                        The shape is [..., K]
                        eig_vectors_middle are the eigen vectors corresponding  eig_values_middle.
                        The shape is [..., K, K]
        """

        derivatives_low = torch.einsum('...bi,...ij,...bj->...b', eig_vectors_low, Gz, eig_vectors_low)
        derivatives_high = torch.einsum('...bi,...ij,...bj->...b', eig_vectors_high, Gz, eig_vectors_high)
        eig_values_estimation = 0.5 * (eig_values_high + eig_values_low) +\
                                    (B_high - B_low) / 8 * (derivatives_high - derivatives_low)
        epsilon = 2 * (eig_values_estimation - eig_values_middle).abs().max(dim=-1)[0]
        return epsilon


    Hamiltonians = torch.stack((F + Gz * B_low, F + Gz * B_high), dim=-3)
    eig_values, eig_vectors = torch.linalg.eigh(Hamiltonians)
    eig_values_low, eig_values_high = eig_values[..., 0, :], eig_values[..., 1, :]
    eig_vectors_low, eig_vectors_high = eig_vectors[..., 0, :, :], eig_vectors[..., 1, :, :]

    intervals_B = [(B_low, B_high)]
    intervals_values = [(eig_values_low, eig_values_high)]
    intervals_vectors = [(eig_vectors_low, eig_vectors_high)]
    iterations = 0

    bisection_mask = tester(eig_values_low, eig_values_high, B_low, B_high)  # True means that continue divide
    bisection_masks = [bisection_mask]
    if torch.all(~bisection_mask):
        warnings.warn("There are no resonance in the interval")
        return None
    final_intervals_B = []
    final_intervals_values = []
    final_intervals_vectors = []
    final_bisection_masks = []
    while intervals_B and iterations < max_iterations:
        new_intervals_B = []
        new_intervals_values = []
        new_intervals_vectors = []
        new_bisection_masks = []
        for (B_low, B_high), (eig_values_low, eig_values_high),\
                (eig_vectors_low, eig_vectors_high), bisection_mask in zip(
                intervals_B, intervals_values, intervals_vectors, bisection_masks):

            B_middle = (B_low + B_high) / 2
            eig_values_middle, eig_vectors_middle = torch.linalg.eigh(F + Gz * B_middle)
            error = compute_error(eig_values_low[bisection_mask],
                                    eig_values_high[bisection_mask], eig_values_middle[bisection_mask],
                                    eig_vectors_low[bisection_mask], eig_vectors_high[bisection_mask],
                                    B_low[bisection_mask], B_high[bisection_mask]
                                    )
            if error.max() < a_tol:
                final_intervals_B.append((B_low, B_high))
                final_intervals_values.append((eig_values_low, eig_values_high))
                final_intervals_vectors.append((eig_vectors_low, eig_vectors_high))
                final_bisection_masks.append(bisection_mask)
            else:
                bisection_mask_low = tester(eig_values_low, eig_values_middle, B_low, B_middle)
                bisection_mask_high = tester(eig_values_middle, eig_values_high, B_middle, B_high)

                if torch.any(bisection_mask_low):
                    new_intervals_B.append((B_low, B_middle))
                    new_intervals_values.append((eig_values_low, eig_values_middle))
                    new_intervals_vectors.append((eig_vectors_low, eig_vectors_middle))
                    new_bisection_masks.append(bisection_mask_low)

                elif torch.any(bisection_mask_high):
                    new_intervals_B.append((B_middle, B_high))
                    new_intervals_values.append((eig_values_middle, eig_values_high))
                    new_intervals_vectors.append((eig_vectors_middle, eig_vectors_high))
                    new_bisection_masks.append(bisection_mask_high)
                else:
                    pass

        intervals_B = new_intervals_B
        intervals_values = new_intervals_values
        intervals_vectors = new_intervals_vectors
        bisection_masks = new_bisection_masks
        iterations += 1

    print(final_intervals_B)

