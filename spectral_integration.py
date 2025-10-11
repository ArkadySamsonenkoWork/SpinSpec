from abc import ABC, abstractmethod
import math
import typing as tp

import torch
import torch.nn as nn

class BaseIntegrand(nn.Module, ABC):
    def _sum_method_fabric(self, harmonic: int = 0) -> tp.Callable[[tp.Any, tp.Any], torch.Tensor]:
        if harmonic == 0:
            return self._absorption
        elif harmonic == 1:
            return self._derivative
        else:
            raise ValueError("Harmonic must be 0 or 1")

    @abstractmethod
    def _absorption(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def _derivative(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass


class EasySpinIntegrand(BaseIntegrand):
    """
    Calculates the term like in EasySpin article
    """

    def __init__(self, harmonic: int, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.sum_method = self._sum_method_fabric(harmonic)
        self.register_buffer("pi_sqrt", torch.tensor(math.sqrt(math.pi), device=device))
        self.register_buffer("two", torch.tensor(2.0, device=device))
        self.register_buffer("cutoff", torch.tensor(3.0, device=device))

    def _absorption(self, arg: torch.Tensor, c_val: torch.Tensor):
        return torch.exp(-arg.square()) * c_val / self.pi_sqrt

    def _derivative(self, arg: torch.Tensor, c_val: torch.Tensor):
        return self.two * arg * torch.exp(-arg.square()) * c_val * c_val / self.pi_sqrt

    def forward(self, B_mean: torch.Tensor, c_extended: torch.Tensor, B_val: torch.Tensor):
        arg = (B_mean - B_val) * c_extended
        out = torch.zeros_like(arg)
        mask = arg.abs() <= self.cutoff
        c_extended = c_extended.expand_as(arg)
        out[mask] = self.sum_method(arg[mask], c_extended[mask])
        return out


class BaseSpectraIntegrator(nn.Module):
    def __init__(self, harmonic: int = 1, natural_width: float = 1e-6, chunk_size=128,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
                 ):
        super().__init__()
        self.harmonic = harmonic

        self.register_buffer("natural_width", torch.tensor(natural_width, device=device, dtype=dtype))
        self.chunk_size = chunk_size
        self.infty_ratio = EasySpinIntegrand(harmonic, device=device)

        self.register_buffer("pi_sqrt", torch.tensor(math.sqrt(math.pi), device=device, dtype=dtype))
        self.register_buffer("two_sqrt", torch.tensor(math.sqrt(2.0), device=device, dtype=dtype))
        self.register_buffer("three", torch.tensor(3.0, device=device, dtype=dtype))
        self.register_buffer("width_conversion", torch.tensor(1/9, device=device, dtype=dtype))
        self.register_buffer("additional_factor", torch.tensor(1.0, device=device, dtype=dtype))

    @abstractmethod
    def forward(self, res_fields: torch.Tensor,
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


class SpectraIntegratorEasySpinLike(BaseSpectraIntegrator):
    def __init__(self, harmonic: int = 1, natural_width: float = 1e-5, chunk_size=128,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        super().__init__(harmonic, natural_width, chunk_size, device=device, dtype=dtype)

    def forward(self, res_fields: torch.Tensor,
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

        spectral_width = spectral_field[..., 1] - spectral_field[..., 0]
        A_mean = A_mean * area

        width = torch.where(width > self.natural_width, width, width + self.natural_width)
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)

        d13 = (B1 - B3) / width
        d23 = (B2 - B3) / width
        d12 = (B1 - B2) / width

        additional_width_square = ((d13.square() + d23.square() + d12.square()) * self.width_conversion)

        width = torch.where(width > 2 * self.natural_width, width, width + 2 * self.natural_width)
        if width.shape[:-1]:
            spectral_width = spectral_width.unsqueeze(-1)
        else:
            spectral_width = spectral_width
        extended_width = (width.square() * (1 + additional_width_square) + spectral_width.square()).sqrt()

        extended_width = torch.where(extended_width < spectral_width, extended_width, extended_width + spectral_width)


        B_mean = ((B1 + B2 + B3) / self.three).unsqueeze(-2)
        c_extended = (self.two_sqrt / extended_width).unsqueeze(-2)
        A_mean = A_mean.unsqueeze(-2)

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
    def __init__(self, harmonic: int = 1, natural_width: float = 1e-6, chunk_size=128,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        super().__init__(harmonic, natural_width, chunk_size, device=device, dtype=dtype)
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        self.register_buffer("two", torch.tensor(2.0, device=device, dtype=dtype))


    def forward(self, res_fields: torch.Tensor,
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
    def forward(self, res_fields: torch.Tensor,
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
        A_mean = (A_mean * area).unsqueeze(-2)


        width = width
        width = self.natural_width + width
        res_fields, _ = torch.sort(res_fields, dim=-1, descending=True)
        B1, B2, B3 = torch.unbind(res_fields, dim=-1)


        d13 = (B1 - B3) / width
        d23 = (B2 - B3) / width
        d12 = (B1 - B2) / width

        additional_width_square = ((d13.square() + d23.square() + d12.square()) * self.width_conversion)

        extended_width = width * (1 + additional_width_square).sqrt()
        B_mean = (B1 + B2 + B3) / self.three
        c_extended = self.two_sqrt / extended_width


        def integrand(B_val: torch.Tensor):
            """
            :param B_val: the value of  spectral magnetic field
            :return: The total intensity at this magnetic field
            """

            ratio = self.infty_ratio(B_mean, c_extended, B_val)
            weighted_ratio = ratio.unsqueeze(-3) * A_mean
            return weighted_ratio.sum(dim=-1)

        chunks = spectral_field.split(self.chunk_size, dim=-1)
        result = torch.cat([integrand(ch.unsqueeze(-1)) for ch in chunks], dim=-1)
        return result


class MeanIntegrator(BaseSpectraIntegrator):
    def __init__(self, harmonic: int = 1, natural_width: float = 1e-4, chunk_size: int = 128,
                 device: torch.device = torch.device("cpu")):
        """
        :param harmonic: The harmonic of the spectra. 0 is an absorptions, 1 is derivative
        """
        super().__init__(harmonic, natural_width, chunk_size, device=device)
        self.infty_ratio = EasySpinIntegrand(harmonic)

    def forward(self, res_fields: torch.Tensor,
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
