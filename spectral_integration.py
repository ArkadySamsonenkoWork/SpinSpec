from abc import ABC, abstractmethod
import math

import torch


class BaseIntegrand(ABC):
    def _sum_method_fabric(self, harmonic: int = 0):
        if harmonic == 0:
            return self._absorption
        elif harmonic == 1:
            return self._derivative
        else:
            raise ValueError("Harmonic must be 0 or 1")

    @abstractmethod
    def _absorption(self, *args, **kwargs):
        pass

    @abstractmethod
    def _derivative(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class EasySpinIntegrand(BaseIntegrand):
    """
    Calculates the term like in EasySpin article
    """

    def __init__(self, harmonic: int):
        self.sum_method = self._sum_method_fabric(harmonic)
        self.pi_sqrt = torch.tensor(math.sqrt(math.pi))
        self.two = torch.tensor(2.0)
        self._arg = torch.tensor(0.0)

    def _absorption(self, arg: torch.Tensor, c_val: torch.Tensor):
        return torch.exp(-arg.square()) * c_val / self.pi_sqrt

    def _derivative(self, arg: torch.Tensor, c_val: torch.Tensor):
        return self.two * arg * torch.exp(-arg.square()) * c_val * c_val / self.pi_sqrt

    def __call__(self, B_mean, c_extended, B_val):
        self._arg = (B_mean - B_val) * c_extended
        ratio = self.sum_method(self.arg, c_extended)
        return ratio

    @property
    def arg(self):
        return self._arg


class AnalyticIntegrand(BaseIntegrand):
    """
    Calculates the term using fully analytical approach
    """

    def __init__(self, harmonic: int):
        self.sum_method = self._sum_method_fabric(harmonic)
        self.pi_sqrt = torch.tensor(math.sqrt(math.pi))
        self.two = torch.tensor(2.0)
        self.eps_val = torch.tensor(1e-10)

    def _absorption(self, arg: torch.Tensor, c_val: torch.Tensor):
        erf_val = torch.erf(arg)
        exp_val = torch.exp(-arg.square())
        return (c_val / self.two) * (arg * erf_val + (1 / self.pi_sqrt) * exp_val)

    def _derivative(self, arg: torch.Tensor, c_val: torch.Tensor):
        erf_val = torch.erf(arg)
        return -(c_val / self.two) * erf_val * c_val

    def __call__(self, B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor, d12: torch.Tensor,
                 d23: torch.Tensor, d13: torch.Tensor, c: torch.Tensor, denominator: torch.Tensor, B_val: torch.Tensor):
        arg_1 = (B1 - B_val) * c
        arg_2 = (B2 - B_val) * c
        arg_3 = (B3 - B_val) * c

        X1 = self.sum_method(arg_1, c)
        X2 = self.sum_method(arg_2, c)
        X3 = self.sum_method(arg_3, c)

        num = (X1 * d23 - X2 * d13 + X3 * d12)
        log_num = torch.log(torch.abs(num) + self.eps_val)
        log_den = torch.log(torch.abs(denominator) + self.eps_val)
        ratio = torch.sign(num) * torch.exp(log_num - log_den)
        return ratio


class BaseSpectraIntegrator:
    @abstractmethod
    def __init__(self, harmonic: int = 0, natural_width: float = 1e-5):
        pass

    @abstractmethod
    def integrate(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean: torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        """
        :param res_fields: The resonance fields with the shape [..., M, 3]
        :param width: The width of transitions. The shape is [..., M]
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param spectral_field: The magnetic fields where spectra should be created. The shape is [...., N]
        :return: result: Tensor of shape (..., N) with the value of the integral for each B
         """
        pass

class SpectraIntegratorExtended(BaseSpectraIntegrator):
    def __init__(self, harmonic: int = 1, natural_width: float = 1e-5):
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        self.pi_sqrt = torch.tensor(math.sqrt(math.pi))
        self.two_sqrt = torch.tensor(math.sqrt(2.0))
        self.natural_width = torch.tensor(natural_width)

        self.criteria_coeff = torch.tensor(4)
        self.three = torch.tensor(3)
        self.shift_coeff = torch.tensor(0.01)
        self.analytical_ratio = AnalyticIntegrand(harmonic)
        self.infty_ratio = EasySpinIntegrand(harmonic)
        self.nine = torch.tensor(9.0)


    def _criteria(self, arg_infty: torch.Tensor, additional_width: torch.Tensor):
        """
        :param arg_infty: the argument of easyspin-like integral function. (B_mean-B_val) * sqrt(2) / width
        :param additional_width: The additional width due to the not equal fields at vertices of triangle
        :return: the criteria. If it is True arg_infty >> additional_width and EasySpin like intensity can be used
        """
        return (arg_infty.abs() / additional_width) < self.criteria_coeff


    def integrate(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean: torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        r"""
        Computes the integral
            I(B) = 1/2 sqrt(2/pi) * (1/width) * A_mean * I_triangle(B) * area,

            at large B because of the instability of analytical solution we use easyspin-like solution with
            effective width
            w_additional = (((B1 - B2)**2 + (B2 - B3)**2 + (B1 - B3)**2) / 9).sqrt()
            w_effective = (w**2 + w_additional**2).sqrt()
        where
        :param res_fields: The resonance fields with the shape [..., M, 3]
        :param width: The width of transitions. The shape is [..., M]
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param spectral_field: The magnetic fields where spectra should be created. The shape is [...., N]
        :return: result: Tensor of shape (..., N) with the value of the integral for each B

        """
        A_mean = A_mean * area

        width = width
        width = self.natural_width + width
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)
        B1 = torch.where((B1 - B2) > width * self.shift_coeff, B1, B1 + width * self.shift_coeff)
        B3 = torch.where((B2 - B3) > width * self.shift_coeff, B1, B3 - width * self.shift_coeff)

        d13 = (B1 - B3) / width
        d23 = (B2 - B3) / width
        d12 = (B1 - B2) / width
        c = self.two_sqrt / width
        denominator = d12 * d23 * d13
        additional_width_square = ((d13.square() + d23.square() + d12.square()) / self.nine)
        additional_width = additional_width_square.sqrt()
        extended_width = width * (1 + additional_width_square).sqrt()
        B_mean = (B1 + B2 + B3) / self.three
        c_extended = self.two_sqrt / extended_width

        def integrand(B_val: torch.Tensor):
            """
            :param B_val: the value of  spectral magnetic field
            :return: The total intensity at this magnetic field
            """
            analytical_ratio = self.analytical_ratio(B1, B2, B3, d12, d23, d13, c, denominator, B_val)
            infty_ratio = self.infty_ratio(B_mean, c_extended, B_val)
            criteria = self._criteria(self.infty_ratio.arg, additional_width)
            ratio = torch.where(criteria, analytical_ratio, infty_ratio)
            return (ratio * A_mean).sum(dim=-1)

        spectral_field = spectral_field.unsqueeze(-1)
        #result = torch.vmap(integrand)(spectral_field)  # To make full integral equel to 1.0
        result = integrand(spectral_field)
        return result


# IT MUST BE CORRECTED TO SPEED UP COMPUTATIONS
class SpectraIntegratorEasySpinLike(BaseSpectraIntegrator):
    def __init__(self, harmonic: int = 1, natural_width: float = 1e-5):
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        self.pi_sqrt = torch.tensor(math.sqrt(math.pi))
        self.two_sqrt = torch.tensor(math.sqrt(2.0))
        self.natural_width = torch.tensor(natural_width)

        self.three = torch.tensor(3)
        self.infty_ratio = EasySpinIntegrand(harmonic)
        self.nine = torch.tensor(9.0)



    def integrate(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean: torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        r"""
        Computes the integral
            I(B) = 1/2 sqrt(2/pi) * (1/width) * A_mean * I_triangle(B) * area,

            at large B because of the instability of analytical solution we use easyspin-like solution with
            effective width
            w_additional = (((B1 - B2)**2 + (B2 - B3)**2 + (B1 - B3)**2) / 9).sqrt()
            w_effective = (w**2 + w_additional**2).sqrt()
        where
        :param res_fields: The resonance fields with the shape [..., M, 3]
        :param width: The width of transitions. The shape is [..., M]
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param spectral_field: The magnetic fields where spectra should be created. The shape is [...., N]
        :return: result: Tensor of shape (..., N) with the value of the integral for each B

        """
        A_mean = A_mean * area

        width = width
        width = self.natural_width + width
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)


        d13 = (B1 - B3) / width
        d23 = (B2 - B3) / width
        d12 = (B1 - B2) / width

        additional_width_square = ((d13.square() + d23.square() + d12.square()) / self.nine)

        extended_width = width * (1 + additional_width_square).sqrt()
        B_mean = (B1 + B2 + B3) / self.three
        c_extended = self.two_sqrt / extended_width

        def integrand(B_val: torch.Tensor):
            """
            :param B_val: the value of  spectral magnetic field
            :return: The total intensity at this magnetic field
            """
            ratio = self.infty_ratio(B_mean, c_extended, B_val)
            return (ratio * A_mean).sum(dim=-1)

        #spectral_field = spectral_field.unsqueeze(-1)
        #result = integrand(spectral_field)
        #result = torch.vmap(integrand)(spectral_field)  # To make full integral equel to 1.0
        result = torch.tensor([integrand(b_val) for b_val in spectral_field])
        return result


class SpectraIntegratorEasySpinLikeTimeResolved(SpectraIntegratorExtended):
    def integrate(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean: torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        r"""
        Computes the integral
            I(B) = 1/2 sqrt(2/pi) * (1/width) * A_mean * I_triangle(B) * area,

            at large B because of the instability of analytical solution we use easyspin-like solution with
            effective width
            w_additional = (((B1 - B2)**2 + (B2 - B3)**2 + (B1 - B3)**2) / 9).sqrt()
            w_effective = (w**2 + w_additional**2).sqrt()
        where
        :param res_fields: The resonance fields with the shape [..., M, 3]
        :param width: The width of transitions. The shape is [..., M]
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param spectral_field: The magnetic fields where spectra should be created. The shape is [...., N]
        :return: result: Tensor of shape (..., N) with the value of the integral for each B

        """
        area = area.unsqueeze(0)
        A_mean = A_mean * area

        width = width
        width = self.natural_width + width
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)


        d13 = (B1 - B3) / width
        d23 = (B2 - B3) / width
        d12 = (B1 - B2) / width

        additional_width_square = ((d13.square() + d23.square() + d12.square()) / self.nine)

        extended_width = width * (1 + additional_width_square).sqrt()
        B_mean = (B1 + B2 + B3) / self.three
        c_extended = self.two_sqrt / extended_width
        c_extended = c_extended.unsqueeze(0)
        B_mean = B_mean.unsqueeze(0)
        spectral_field = spectral_field.unsqueeze(-1)
        def integrand(B_val: torch.Tensor):
            """
            :param B_val: the value of  spectral magnetic field
            :return: The total intensity at this magnetic field
            """
            ratio = self.infty_ratio(B_mean, c_extended, B_val)
            return (ratio * A_mean).sum(dim=-1)

        #spectral_field = spectral_field.unsqueeze(-1)
        #result = integrand(spectral_field)
        #result = torch.vmap(integrand)(spectral_field)  # To make full integral equel to 1.0
        result = torch.stack([integrand(b_val) for b_val in spectral_field], dim=0)
        return result



# Do NOT WORK. NOT IMPLEMENTED
class SpectraIntegratorFullyAnalytical(BaseSpectraIntegrator):
    def __init__(self, harmonic: int = 0):
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        self.pi_sqrt = torch.tensor(math.sqrt(math.pi))
        self.two_sqrt = torch.tensor(math.sqrt(2.0))

        self.natural_width = 1e-5
        self.eps_val = torch.tensor(1e-12)
        self.threshold = torch.tensor(1e-6)
        self.clamp = torch.tensor(50)
        self.sum_method = self._sum_method_fabric(harmonic)
        self.equality_threshold = 1e-2
        raise NotImplementedError


    def _sum_method_fabric(self, harmonic: int = 0):
        if harmonic == 0:
            return self._absorption
        elif harmonic == 1:
            return self._derivative
        else:
            raise ValueError("Harmonic must be 0 or 1")

    def _absorption(self, x: torch.Tensor, c_val: torch.Tensor, width_val: torch.Tensor):
        arg = c_val * x
        erf_val = torch.erf(arg)
        exp_val = torch.exp(torch.clamp(-arg.square(), min=-50))  # Prevent underflow
        return (arg * erf_val + (1 / self.pi_sqrt) * exp_val)


    def _derivative(self, x: torch.Tensor, c_val: torch.Tensor, width_val: torch.Tensor):
        arg = torch.clamp(c_val * x, -self.clamp, self.clamp)
        return torch.erf(arg)

    def _get_equality_masks(self, B1, B2, B3, width):
        diff12 = torch.abs(B1 - B2) / width
        diff13 = torch.abs(B1 - B3) / width
        diff23 = torch.abs(B2 - B3) / width

        eq12 = diff12 < self.equality_threshold
        eq13 = diff13 < self.equality_threshold
        eq23 = diff23 < self.equality_threshold

        mask_all_diff = ~(eq12 | eq13 | eq23)

        # Case 2: Two are equal, third is different
        mask_two_eq = (eq12 & ~eq13) | (eq13 & ~eq12) | (eq23 & ~eq12)

        # Case 3: All three are equal
        mask_all_eq = eq12 & eq13 & eq23

        return mask_all_diff, mask_two_eq, mask_all_eq

    def _two_equality_case(self, B1, B2, B3, width):
        diff12 = torch.abs(B1 - B2) / width
        diff13 = torch.abs(B1 - B3) / width
        diff23 = torch.abs(B2 - B3) / width

        eq12 = diff12 < self.equality_threshold
        eq13 = diff13 < self.equality_threshold
        eq23 = diff23 < self.equality_threshold

        B1, B3 = torch.where(eq12, B3, B1), torch.where(eq12, B1, B3)
        B1, B2 = torch.where(eq13, B2, B1), torch.where(eq13, B1, B2)

        return B1, B2, B3

    def _apply_mask(self, B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor,
                    width: torch.Tensor, A_mean :torch.Tensor,
                    area: torch.Tensor, mask):
        B1 = B1[mask]
        B2 = B2[mask]
        B3 = B3[mask]
        area = area[mask]
        A_mean = A_mean[mask]
        width = width[mask]
        return B1, B2, B3, width, area, A_mean


    def all_different_case(self, B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor,
                           width: torch.Tensor, A_mean :torch.Tensor,
                           area: torch.Tensor, spectral_field: torch.Tensor, mask_all_diff):
        B1, B2, B3, width, area, A_mean = self._apply_mask(B1, B2, B3, width, A_mean, area, mask_all_diff)

        d13 = (B1 - B3) / width
        d23 = (B2 - B3) / width
        d12 = (B1 - B2) / width

        c = self.two_sqrt / width
        denominator = d12 * d23 * d13
        safe_denom = torch.where(torch.abs(denominator) < self.threshold,
                                 denominator + self.eps_val,
                                 denominator)
        def integrand(B_val):
            arg_1 = (B1 - B_val) / width
            arg_2 = (B2 - B_val) / width
            arg_3 = (B3 - B_val) / width
            X1 = self.sum_method(B1 - B_val, c, width)
            X2 = self.sum_method(B2 - B_val, c, width)
            X3 = self.sum_method(B3 - B_val, c, width)

            num = (X1 * d23 - X2 * d13 + X3 * d12)
            # num = torch.where((arg_1.square() + arg_2.square() + arg_3.square()).sqrt() < 20, num, 0.0)  # IT HEALS BEHAVIOUR AT INFTY

            # log_num = torch.log(torch.abs(num) + self.eps_val)
            # log_den = torch.log(torch.abs(safe_denom))
            ratio = num / safe_denom
            # ratio = torch.sign(num) * torch.exp(log_num - log_den)
            return (ratio * (1 / self.two_sqrt) * (A_mean * area) * (1 / width)).sum(dim=-1)

        result = torch.vmap(integrand)(spectral_field)
        return result


    def two_eq_case(self, B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor,
                    width: torch.Tensor, A_mean: torch.Tensor,
                    area: torch.Tensor, spectral_field: torch.Tensor, mask_two_equel):
        B1, B2, B3, width, area, A_mean = self._apply_mask(B1, B2, B3, width, A_mean, area, mask_two_equel)
        B1, B2, B3 = self._two_equality_case(B1, B2, B3, width)  # B2 == B3

        d13 = (B1 - B3) / width
        d12 = (B1 - B2) / width

        c = self.two_sqrt / width
        denominator = d12 * d13

        safe_denom = torch.where(torch.abs(denominator) < self.threshold,
                                 denominator + self.eps_val,
                                 denominator)

        def F(x1, x2, c_val, width):
            arg = torch.clamp(c_val * x2, -self.clamp, self.clamp)  # Prevent large values
            erf_val = torch.erf(arg)
            exp_val = torch.exp(torch.clamp(-arg.square(), min=-50))  # Prevent underflow
            return (c_val * x1 * erf_val + (1 / self.pi_sqrt) * exp_val)

        def integrand(B_val):
            X1 = F(B1 - B_val, B1 - B_val, c, width)
            X2 = F(B1 - B_val, B2 - B_val, c, width)
            num = X1 - X2

            eps = self.eps_val
            log_num = torch.log(torch.abs(num) + eps)
            log_den = torch.log(torch.abs(safe_denom))

            ratio = torch.sign(num) * torch.exp(log_num - log_den)
            return (ratio * (1 / (2 * self.two_sqrt)) * (A_mean * area) * (1 / width)).sum(dim=-1)
        result = torch.vmap(integrand)(spectral_field)
        return result


    def all_eq_case(self, B1: torch.Tensor, B2: torch.Tensor, B3: torch.Tensor,
                    width: torch.Tensor, A_mean: torch.Tensor,
                    area: torch.Tensor, spectral_field: torch.Tensor, mask_all_equel):
        B1, B2, B3, width, area, A_mean = self._apply_mask(B1, B2, B3, width, A_mean, area, mask_all_equel)
        B0 = (B1 + B2 + B3) / 3
        d23 = (B2 - B3)
        d13 = (B1 - B3)
        d12 = (B1 - B2)
        width = (width.square() + (d12.square() + d13.square() + d23.square()) / 9).sqrt()
        c = self.two_sqrt / width
        def integrand(B_val):
            arg = (B0 - B_val) * c
            term = (self.two_sqrt / self.pi_sqrt) * torch.exp(-arg.square()) * (1 / width)
            return (term * (A_mean * area)).sum(dim=-1)

        result = torch.vmap(integrand)(spectral_field)
        return result

    def integrate(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean :torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        r"""
        Computes the integral
            I(B) = sqrt(2/pi) * (1/width) * A_mean * I_triangle(B) * area,
        where
        :param res_fields: The resonance fields with the shape [..., M, 3]
        :param width: The width of transitions. The shape is [..., M]
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param spectral_field: The magnetic fields where spectra should be created. The shape is [...., N]
        :return: result: Tensor of shape (..., N) with the value of the integral for each B
        """
        width = width
        width = self.natural_width + width
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)

        mask_all_diff, mask_two_eq, mask_all_eq = self._get_equality_masks(B1, B2, B3, width)


        result = \
            self.all_different_case(B1, B2, B3, width, A_mean, area, spectral_field, mask_all_diff) + \
            self.two_eq_case(B1, B2, B3, width, A_mean, area, spectral_field, mask_two_eq) + \
            self.all_eq_case(B1, B2, B3, width, A_mean, area, spectral_field, mask_all_eq)

        return result