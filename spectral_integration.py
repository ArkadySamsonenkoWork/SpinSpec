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
        self.cutoff = torch.tensor(3.0)

    def _absorption(self, arg: torch.Tensor, c_val: torch.Tensor):
        return torch.exp(-arg.square()) * c_val / self.pi_sqrt

    def _derivative(self, arg: torch.Tensor, c_val: torch.Tensor):
        return self.two * arg * torch.exp(-arg.square()) * c_val * c_val / self.pi_sqrt

    def __call__(self, B_mean, c_extended, B_val):
        arg = (B_mean - B_val) * c_extended
        out = torch.zeros_like(arg)
        mask = arg.abs() <= self.cutoff
        c_extended = c_extended.expand_as(arg)
        out[mask] = self.sum_method(arg[mask], c_extended[mask])
        return out


class BaseSpectraIntegrator:
    def __init__(self, harmonic: int = 1, natural_width: float = 1e-4, chunk_size=128):
        self.harmonic = harmonic
        self.natural_width = natural_width
        self.chunk_size = chunk_size

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


# IT MUST BE CORRECTED TO SPEED UP COMPUTATIONS
class SpectraIntegratorEasySpinLike(BaseSpectraIntegrator):
    def __init__(self, harmonic: int = 1, natural_width: float = 1e-4, chunk_size=128):
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        super().__init__(harmonic, natural_width, chunk_size)
        self.pi_sqrt = torch.tensor(math.sqrt(math.pi))
        self.two_sqrt = torch.tensor(math.sqrt(2.0))

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

        chunks = spectral_field.split(self.chunk_size, dim=-1)
        result = torch.cat([integrand(ch.unsqueeze(-1)) for ch in chunks], dim=-1)
        return result


class AxialSpectraIntegratorEasySpinLike(SpectraIntegratorEasySpinLike):
    def __init__(self, harmonic: int = 1, natural_width: float = 1e-4, chunk_size=128):
        super().__init__(harmonic, natural_width, chunk_size)
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        self.two = torch.tensor(2)


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
        width = self.natural_width + width
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2 = torch.unbind(res_fields, dim=-1)

        d12 = (B1 - B2) / width

        additional_width_square = d12.square() / self.three
        extended_width = width * (1 + additional_width_square).sqrt()
        B_mean = (B1 + B2) / self.two
        c_extended = self.two_sqrt / extended_width

        def integrand(B_val: torch.Tensor):
            """
            :param B_val: the value of  spectral magnetic field
            :return: The total intensity at this magnetic field
            """
            ratio = self.infty_ratio(B_mean, c_extended, B_val)
            weighted_ratio = ratio * A_mean
            return weighted_ratio.sum(dim=1)

        chunks = spectral_field.split(self.chunk_size, dim=-1)
        result = torch.cat([integrand(ch.unsqueeze(-1)) for ch in chunks], dim=-1)
        return result


class SpectraIntegratorEasySpinLikeTimeResolved(SpectraIntegratorEasySpinLike):
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
            arg = (B_mean - B_val) * c_extended
            ratio = torch.exp(-arg.square()) * c_extended / self.pi_sqrt
            return (ratio * A_mean).sum(dim=-1)

        result = torch.stack([integrand(b_val) for b_val in spectral_field], dim=0)
        return result


class MeanIntegrator(BaseSpectraIntegrator):
    def __init__(self, harmonic: int = 1, natural_width: float = 1e-4, chunk_size: int = 128):
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        super().__init__(harmonic, natural_width, chunk_size)
        self.pi_sqrt = torch.tensor(math.sqrt(math.pi))
        self.two_sqrt = torch.tensor(math.sqrt(2.0))

        self.three = torch.tensor(3)
        self.nine = torch.tensor(9.0)
        self.infty_ratio = EasySpinIntegrand(harmonic)

    def integrate(self, res_fields: torch.Tensor,
                  width: torch.Tensor, A_mean: torch.Tensor,
                  area: torch.Tensor, spectral_field: torch.Tensor):
        r"""
        Computes the mean intensity
        :param res_fields: The resonance fields with the shape [..., M]
        :param width: The width of transitions. The shape is [..., M]
        :param A_mean: The intensities of transitions. The shape is [..., M]
        :param area: The area of transitions. The shape is [M]. It is the same for all batch dimensions
        :param spectral_field: The magnetic fields where spectra should be created. The shape is [...., N]
        :return: result: Tensor of shape (..., N) with the value of the integral for each B

        """
        A_mean = A_mean * area
        width = width
        width = self.natural_width + width

        c_extended = self.two_sqrt / width

        def integrand(B_val: torch.Tensor):
            """
            :param B_val: the value of  spectral magnetic field
            :return: The total intensity at this magnetic field
            """
            ratio = self.infty_ratio(res_fields, c_extended, B_val)
            return (ratio * A_mean).sum(dim=-1)

        chunks = spectral_field.split(self.chunk_size, dim=-1)
        result = torch.cat([integrand(ch.unsqueeze(-1)) for ch in chunks], dim=-1)
        return result
