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

    K = eig_values_low.shape[-1]  # Number of states
    u, v = torch.triu_indices(K, K, offset=1, device=eig_values_low.device)


    res_low =\
        (eig_values_low.unsqueeze(-2)[..., v] - eig_values_low.unsqueeze(-2)[..., u]).squeeze(-2) - resonance_frequency
    res_high = \
        (eig_values_high.unsqueeze(-2)[..., v] - eig_values_high.unsqueeze(-2)[..., u]).squeeze(-2) - resonance_frequency

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
    #mask = torch.any(mask, dim=(-1, -2))
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


def has_validity_sign_rule(B_low, B_high, deriv_max, baseline_sign_mask, res_low, res_high):
    rapid_variation = has_rapid_variation(res_low, res_high, deriv_max, B_low, B_high)
    sign_change = has_sign_change(res_low, res_high)
    mask_delta = torch.where(condition=baseline_sign_mask, input=rapid_variation, other=sign_change)
    return mask_delta

def check_resonance(eig_values_low: torch.Tensor, eig_values_high: torch.Tensor,
                    B_low: torch.Tensor, B_high: torch.Tensor,
                    deriv_max: torch.Tensor, baseline_sign_mask: torch.Tensor,
                    resonance_frequency: torch.Tensor
                    ):
    """
    Compute the error after division of the interval
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
    mask_monotonically = has_monotonically_rule(
        eig_values_low, eig_values_high, resonance_frequency)  # [...]

    res_low, res_high = compute_resonance_functions(
        eig_values_low, eig_values_high, resonance_frequency)

    mask_delta = has_rapid_variation(res_low, res_high, deriv_max, B_low, B_high)
    mask_sign_change = has_sign_change(res_low, res_high)

    return torch.logical_and(mask_monotonically, torch.where(baseline_sign_mask, mask_delta, mask_sign_change))

def compute_derivative(eigen_vector, G):
    return torch.einsum('...bi,...ij,...bj->...b', torch.conj(eigen_vector), G, eigen_vector).real

# Must me rebuild to speed up.
# 1) After each while iteration, it is possible to stack intervals to make bigger batches
# 2) Also, it is possible to stack all lists of tensor to one stack to increase the speed.
# 3) Maybe, it is better to avoid storing of deriv_max at the list and use indexes every time
# 4) converged_mask.any(). I have calculated the eigen_val and eigen_vec at the middle magnetic field.
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


def get_resonance_intervals(F: torch.Tensor, Gz: torch.Tensor,
                            B_low: torch.Tensor, B_high: torch.Tensor,
                            deriv_max: torch.Tensor, resonance_frequency: torch.Tensor):
    """
    Calculate the resonance intervals, where the resonance field is possible
    :param F: Magnetic filed free stationary Hamiltonian matrix. The shape is [..., K, K],
    where K is spin system dimension
    :param Gz: Magnetic field dependant part of stationary Hamiltonian with the shape [..., K, K].
    :param B_low: The start of the interval to find roots. The shape is [...]
    :param B_high: The end of the interval to find roots. The shape is [...]
    :param deriv_max: The maximum value of the energy derivatives. The shape is [...]
    :param resonance_frequency: The resonance frequency. The shape is []
    :return:
    """
    r_tol = 1e-5
    max_iterations = 100
    a_tol = resonance_frequency * r_tol

    def compute_error(eig_values_low: torch.Tensor,
                      eig_values_high: torch.Tensor,
                      eig_values_middle: torch.Tensor,
                      eig_vectors_low: torch.Tensor,
                      eig_vectors_high: torch.Tensor,
                      B_low: torch.Tensor, B_high: torch.Tensor,
                      indexes: torch.Tensor):
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
        :param B_low: The lower magnetic field The shape is [..., 1, 1]
        :param B_high: The higher magnetic field The shape is [..., 1, 1]
        :param indexes: Indexes where Gz must be slised. The bool tensor with the shape of the initial shape [...]
        :return: epsilon is epsilon mistake. The tensor with the shape [...]
        """

        derivatives_low = compute_derivative(eig_vectors_low, Gz[indexes])
        derivatives_high = compute_derivative(eig_vectors_high, Gz[indexes])
        eig_values_estimation = 0.5 * (eig_values_high + eig_values_low) +\
                                    (B_high - B_low) / 8 * (derivatives_high - derivatives_low)
        epsilon = 2 * (eig_values_estimation - eig_values_middle).abs().max(dim=-1)[0]
        return epsilon, (derivatives_low, derivatives_high)

    def update_batch(batches: list[dict], eig_values_low, eig_values_mid, eig_values_high,
                     eig_vectors_low, eig_vectors_mid, eig_vectors_high,
                     B_low, B_mid, B_high,
                     deriv_max, baseline_sign, indexes,
                     resonance_frequency):

        mask_left = check_resonance(eig_values_low, eig_values_mid, B_low, B_mid,
                                    deriv_max, baseline_sign,
                                    resonance_frequency)
        mask_right = check_resonance(eig_values_mid, eig_values_high, B_mid, B_high,
                                     deriv_max, baseline_sign,
                                     resonance_frequency)

        mask_and = torch.logical_and(mask_left, mask_right)
        mask_xor = torch.logical_xor(mask_left, mask_right)

        # Process and case. It means that both intervals have resonance

        if mask_and.any():
            indexes_and = indexes.clone()
            indexes_and[indexes_and == True] = mask_and

            deriv_max_and = deriv_max[mask_and]
            baseline_sign_and = baseline_sign[mask_and]
            #print(indexes_and)
            batches.append({
                "B": (B_low[mask_and], B_mid[mask_and]),
                "values": (eig_values_low[mask_and], eig_values_mid[mask_and]),
                "vectors": (eig_vectors_low[mask_and], eig_vectors_mid[mask_and]),
                "deriv_max": deriv_max_and,
                "baseline_sign": baseline_sign_and,
                "indexes": indexes_and
            })
            batches.append({
                "B": (B_mid[mask_and], B_high[mask_and]),
                "values": (eig_values_mid[mask_and], eig_values_high[mask_and]),
                "vectors": (eig_vectors_mid[mask_and], eig_vectors_high[mask_and]),
                "deriv_max": deriv_max_and,
                "baseline_sign": baseline_sign_and,
                "indexes": indexes_and
            })

        # Process XOR case. It means that only one interval has resonance.
        # Note, that it is impossible that none interval has resonance

        if mask_xor.any():
            batches.append(
                compute_xor_interval(
                    mask_xor,
                    mask_left,
                    mask_right,
                    (eig_values_low, eig_values_mid, eig_values_high),
                    (eig_vectors_low, eig_vectors_mid, eig_vectors_high),
                    (B_low, B_mid, B_high),
                    deriv_max,
                    baseline_sign,
                    indexes
                )
            )

    def compute_xor_interval(
            bisection_mask_xor: torch.Tensor,
            bisection_mask_left: torch.Tensor,
            bisection_mask_right: torch.Tensor,
            eig_values_data: tuple,
            eig_vectors_data: tuple,
            B_data: tuple,
            deriv_max: torch.Tensor,
            baseline_sign_mask: torch.Tensor,
            indexes: torch.Tensor
    ):
        """Update interval lists for XOR case (resonance in exactly one sub-interval).
            Note, that in this case the resonance will be in ine interval and only theere
        """
        eig_values_low, eig_values_mid, eig_values_high = eig_values_data
        eig_vectors_low, eig_vectors_mid, eig_vectors_high = eig_vectors_data
        B_low, B_mid, B_high = B_data

        # Get adjusted intervals for XOR case
        (B_low, B_high), (eig_values_low, eig_values_high), \
            (eig_vectors_low, eig_vectors_high), deriv_max, \
            baseline_sign, indexes = _handle_xor_case(
            bisection_mask_xor,
            bisection_mask_left,
            bisection_mask_right,
            (eig_values_low, eig_values_mid, eig_values_high),
            (eig_vectors_low, eig_vectors_mid, eig_vectors_high),
            (B_low, B_mid, B_high),
            deriv_max,
            baseline_sign_mask,
            indexes
        )
        new_interval = {"B": (B_low, B_high),
                        "values": (eig_values_low, eig_values_high),
                        "vectors": (eig_vectors_low, eig_vectors_high),
                        "deriv_max": deriv_max,
                        "baseline_sign": baseline_sign,
                        "indexes": indexes,
                        }
        return new_interval

    def _handle_xor_case(
            mask_xor: torch.Tensor,
            mask_left: torch.Tensor,
            mask_right: torch.Tensor,
            eig_values_data: tuple,
            eig_vectors_data: tuple,
            B_data: tuple,
            deriv_max: torch.Tensor,
            baseline_sign: torch.Tensor,
            indexes: torch.Tensor
    ) -> tuple:
        """Process XOR case by mixing the intervals"""

        eig_values_low, eig_values_mid, eig_values_high = eig_values_data
        eig_vectors_low, eig_vectors_mid, eig_vectors_high = eig_vectors_data
        B_low, B_mid, B_high = B_data

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
            (B_low, B_high),
            (eig_values_low, eig_values_high),
            (eig_vectors_low, eig_vectors_high),
            deriv_max[mask_xor],
            baseline_sign[mask_xor],
            indexes
        )
    baseline_sign = get_zero_resonance_baseline(F, resonance_frequency)
    B_low = B_low[..., None, None]
    B_high = B_high[..., None, None]
    Hamiltonians = torch.stack((F + Gz * B_low, F + Gz * B_high), dim=-3)
    eig_values, eig_vectors = torch.linalg.eigh(Hamiltonians)
    eig_values_low, eig_values_high = eig_values[..., 0, :], eig_values[..., 1, :]
    eig_vectors_low, eig_vectors_high = eig_vectors[..., 0, :, :], eig_vectors[..., 1, :, :]

    iterations = 0

    # True means that continue divide
    active_mask = check_resonance(eig_values_low, eig_values_high,
                                  B_low, B_high,
                                  deriv_max, baseline_sign,
                                  resonance_frequency
    )
    if torch.all(~active_mask):
        warnings.warn("There are no resonance in the interval")
        return None
    final_batches = []
    current_batches = [{
        "B": (B_low[active_mask], B_high[active_mask]),
        "values": (eig_values_low[active_mask], eig_values_high[active_mask]),
        "vectors": (eig_vectors_low[active_mask], eig_vectors_high[active_mask]),
        "deriv_max": deriv_max[active_mask],
        "baseline_sign": baseline_sign[active_mask],
        "indexes": active_mask
    }]

    while current_batches and iterations < max_iterations:
        next_batches = []
        for batch in current_batches:
            (B_low, B_high) = batch["B"]
            (eig_values_low, eig_values_high) = batch["values"]
            (eig_vectors_low, eig_vectors_high) = batch["vectors"]
            indexes = batch["indexes"]
            deriv_max = batch["deriv_max"]
            baseline_sign = batch["baseline_sign"]

            B_mid = (B_low + B_high) / 2

            eig_values_mid, eig_vectors_mid = torch.linalg.eigh(F[indexes] + Gz[indexes] * B_mid)
                                                                            # It is only one single
                                                                            # point where gradient should be calculated
            error, (derivatives_low, derivatives_high) =\
                compute_error(eig_values_low, eig_values_high, eig_values_mid,
                            eig_vectors_low, eig_vectors_high,
                            B_low, B_high, indexes
                )

            converged_mask = (error <= a_tol).any(dim=-1)
            # На этом шаге нужно также разделить инетервал на две части. eig_values_mid, eig_vectors_mid уже посчитаны!!
            # Но нужно тогда пересчитывать derivatives
            if converged_mask.any():
                indexes_conv = indexes.clone()
                indexes_conv[indexes_conv == True] = converged_mask
                final_batches.append({
                    "B": (B_low[converged_mask], B_high[converged_mask]),
                    "values": (eig_values_low[converged_mask], eig_values_high[converged_mask]),
                    "vectors": (eig_vectors_low[converged_mask], eig_vectors_high[converged_mask]),
                    "baseline_sign": baseline_sign[converged_mask],
                    "indexes": indexes_conv,
                    "derivatives": (derivatives_low, derivatives_high),
                })

            active_mask = ~converged_mask
            if not active_mask.any():
                continue

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
            #indexes = batch["indexes"]
            indexes[indexes == True] = active_mask

            deriv_max = deriv_max[active_mask]
            baseline_sign = baseline_sign[active_mask]

            update_batch(
                next_batches,
                eig_values_low, eig_values_mid, eig_values_high,
                eig_vectors_low, eig_vectors_mid, eig_vectors_high,
                B_low, B_mid, B_high,
                deriv_max, baseline_sign, indexes,
                resonance_frequency)

        current_batches = next_batches
        iterations += 1
    locate_resonance_fields(final_batches, resonance_frequency)
    return final_batches



# Может это нужно делать через разреженные матрицы.... Я не понимаю...
# Я считаю все произведения U Gx V даже, если переходов нет. Если переходов нет, то вес ноль. Может,
# это стоит оптимизировать и сделать опреацию дешевле и считать только в valid_u и valid_b.
# Но тогда стоятся лишние струкутуры
# Может быть стоит разделять батчи дальше по u и v....
# Дважды считаю коэффициенты полинома. Нужно будет переделать. Один раз для нахождения корней,
# один раз для получения энергий.
# Возможно, стоит избавиться от двух масок.
class ResonanceLocator:
    def __init__(self, max_iterations=10, tolerance=1e-6, accuracy=0.001):
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
        max_iterations = 10
        tolerance = 1e-6  # small value to avoid division by zero

        (coef_3, coef_2, coef_1, coef_0) = self._compute_cubic_polinomial_coeffs(
            diff_eig_low, diff_eig_high, diff_deriv_low, diff_deriv_high)
        coef_0 -= resonance_frequency
        t = - diff_eig_low / (diff_eig_high - diff_eig_low)
        for _ in range(max_iterations):
            poly_val = coef_3 * t ** 3 + coef_2 * t ** 2 + coef_1 * t + coef_0
            poly_deriv = 3 * coef_3 * t ** 2 + 2 * coef_2 * t + coef_1
            t -= poly_val / (poly_deriv + tolerance)
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
                                  diff_deriv_high, mask_cut, resonance_frequency):
        step_B = torch.zeros_like(mask_cut, dtype=torch.float32)
        step_B[mask_cut] = self._find_resonance_field_newton(diff_eig_low, diff_eig_high,
                                                  diff_deriv_low, diff_deriv_high,
                                                  resonance_frequency)
        return step_B


    def _compute_resonance_energies(self, step_B, eig_values_low, eig_values_high, deriv_low, deriv_high):
        step_B = step_B.unsqueeze(-1)
        (coef_3, coef_2, coef_1, coef_0) = self._compute_cubic_polinomial_coeffs(
            eig_values_low, eig_values_high, deriv_low, deriv_high)
        energy = coef_3 * step_B ** 3 + coef_2 * step_B ** 2 + coef_1 * step_B + coef_0
        return energy

    def iterate_over_batche(self, batch, resonance_frequency):
        B_low, B_high = batch["B"]
        eig_values_low, eig_values_high = batch["values"]
        eig_vectors_low, eig_vectors_high = batch["vectors"]
        deriv_low, deriv_high = batch["derivatives"]
        baseline_sign = batch["baseline_sign"]
        indexes = batch["indexes"]
        delta_B = B_high - B_low  # [..., 1, 1]
        shape = eig_values_low.shape  # shape is torch.Size([300, 4])
        K = shape[-1]
        u, v = torch.triu_indices(K, K, offset=1)
        num_pairs = len(u)
        eig_values_low = eig_values_low.unsqueeze(-2)  # [..., 1, K]
        eig_values_high = eig_values_high.unsqueeze(-2)
        deriv_low = deriv_low.unsqueeze(-2)
        deriv_high = deriv_high.unsqueeze(-2)


        diff_eig_low = (eig_values_low[..., v] - eig_values_low[..., u]).squeeze(-2)
        diff_eig_high = (eig_values_high[..., v] - eig_values_high[..., u]).squeeze(-2)

        mask = self.get_resonance_mask(diff_eig_low, diff_eig_high, resonance_frequency)
        mask_triu = mask.any(dim=0)  # For some u and v there are no transitions
        mask_cut = mask[..., mask_triu]

        #valid_indices = torch.nonzero(mask, as_tuple=True)[0]  # Get valid indices

        diff_eig_low = diff_eig_low[mask]
        diff_eig_high = diff_eig_high[mask]

        diff_deriv_low = (delta_B * (deriv_low[..., v] - deriv_low[..., u])).squeeze(-2)[mask]
        diff_deriv_high = (delta_B * (deriv_high[..., v] - deriv_high[..., u])).squeeze(-2)[mask]
        step_B = self._compute_resonance_fields(diff_eig_low, diff_eig_high, diff_deriv_low,
                                    diff_deriv_high, mask_cut, resonance_frequency)
        # Correctly apply mask to u and v
        #u_exp = u.unsqueeze(0).expand(mask.size(0), -1)  # shape: [300, 6]
        #v_exp = v.unsqueeze(0).expand(mask.size(0), -1)  # shape: [300, 6]

        resonance_energies = self._compute_resonance_energies(step_B,
                                                              eig_values_low, eig_values_high, deriv_low, deriv_high)
        valid_u = u[mask_triu]
        valid_v = v[mask_triu]
        weights_low, weights_high = self._compute_linear_interpolation_weights(step_B)
        vectors_u = eig_vectors_low[..., valid_u, :] * weights_low + eig_vectors_high[..., valid_u, :] * weights_high
        vectors_v = eig_vectors_low[..., valid_v, :] * weights_low + eig_vectors_high[..., valid_v, :] * weights_high
        batch["B_res"] = B_low + step_B * delta_B
        batch["energies"] = resonance_energies
        batch["vectors"] = vectors_u, vectors_v
        batch["mask_cut"] = mask_cut
        batch["u"] = valid_u
        batch["v"] = valid_v


# If baseline_sign == True, than only one single root can be in the interval. No looping roots.
# Дальше проверяется, что производная не меняет знак: d <= 0
# Есть doc документ с описанием логики работы
def locate_resonance_fields(final_batches, resonance_frequency):
    max_iterations = 100
    res_loc = ResonanceLocator()
    for batch in final_batches:
        res_loc.iterate_over_batche(batch, resonance_frequency)
    return final_batches






