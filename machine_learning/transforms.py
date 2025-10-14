import sys
import random
import typing as tp
import math

import torch.nn as nn
import torch

sys.path.append("..")

import utils
import constants
import spectra_manager


def vector_to_de(vector: torch.Tensor):
    mean = vector.mean(dim=-1, keepdim=True)
    deviations = vector - mean
    D = (3 / 2) * deviations[..., 2:3]
    E = (deviations[..., 0:1] - deviations[..., 1:2]) / 2
    return torch.cat([mean, D, E], dim=-1)


class ComponentsTransform:
    def __init__(self, freq_factor: float = 200.0, temp_factor: float = 50.0):
        self.freq_factor = torch.tensor(freq_factor)
        self.temp_factor = torch.tensor(temp_factor)

    def __call__(self, components, temperature):
        components = vector_to_de(components)
        components_feat_freq = constants.PLANCK * components / (2.00 * constants.BOHR)
        components_feat_temp = components / constants.unit_converter(temperature, "K_to_Hz").unsqueeze(0)
        return components_feat_freq / self.freq_factor, components_feat_temp / self.temp_factor


class AnglesTransform:
    def __call__(self, angles: torch.Tensor):
        angles = utils.get_canonical_orientations(angles.transpose(0, -2)).transpose(0, -2)
        angles[..., 0] = angles[..., 0] / (2 * torch.pi)
        angles[..., 1] = angles[..., 1] / torch.pi
        angles[..., 2] = angles[..., 2] / (2 * torch.pi)
        return angles


class ComponentsAnglesTransform:
    def __init__(self):
        self.angles_transform = AnglesTransform()
        self.components_transform = ComponentsTransform()

    def __call__(self, components: torch.Tensor, temperature: torch.Tensor, angles: torch.Tensor):
        components_feat_freq, components_feat_temp = self.components_transform(components, temperature)
        angles = self.angles_transform(angles)
        return torch.cat((components_feat_freq, components_feat_temp, angles), dim=-1)


class SpecTransformField:
    def __init__(self, g_tensor_shift: float = 2.0, freq_shift: float = 20.0 * 1e9, freq_deriv: float = 20.0 * 1e9):
        self.g_tensor_shift = torch.tensor(g_tensor_shift)
        self.freq_shift = freq_shift
        self.freq_deriv = freq_deriv

    def __call__(self, field: torch.Tensor, freq: torch.Tensor):

        g_tensors = (constants.PLANCK * freq.unsqueeze(-1)) / (constants.BOHR * field)
        g_tensors = torch.flip(g_tensors, dims=(-1,))
        g_feature = g_tensors - self.g_tensor_shift
        freq_feature = (freq - self.freq_shift) / self.freq_deriv
        return g_feature, freq_feature


class SpecTransformSpecIntensity:
    def __call__(self, spec: torch.Tensor):
        spec = torch.flip(spec, dims=(-1,))
        spec = spec / torch.max(spec, dim=-1, keepdim=True)[0]
        return spec


class SpinTranform:
    def __init__(self, shift: float = 1.0, std: float = 2.0):
        self.shift = torch.tensor(shift)
        self.std = torch.tensor(std)

    def __call__(self, spins: torch.Tensor, types: torch.Tensor):
        spins = (spins - self.shift) / self.std
        return spins


class BroadTransform:
    def __init__(self, shift: float = constants.unit_converter(0.5e-1, "T_to_Hz_e"),
                 std: float = constants.unit_converter(1e-1, "T_to_Hz_e")):
        self.shift = torch.tensor(shift)
        self.std = torch.tensor(std)

    def __call__(self, ham_strain: torch.Tensor, lorentz: torch.Tensor, gauss: torch.Tensor):
        ham_strain = vector_to_de(ham_strain)
        lorentz = lorentz * constants.BOHR / constants.PLANCK
        gauss = gauss * constants.BOHR / constants.PLANCK

        return (torch.cat((ham_strain, lorentz.unsqueeze(-1), gauss.unsqueeze(-1)), dim=-1) - self.shift) / self.std


class SpecFieldPrepare(nn.Module):
    def __init__(self,
                 min_width: float = 1e-4,
                 max_width: float = 1e-1,
                 init_interpolation_points: int = 2000,
                 max_add_points: int = 2000,
                 out_points: int = 1000,
                 spectral_width_factor: float = 4,
                 rng_generator: tp.Optional[random.Random] = None):
        """
        Prepare magnetic field, spectra, Gaussian and Lorentzian tensors using the following procedure:

        1) Each spectrum is interpolated to init_interpolation_points number of points
        2) Additional points are added to each side of the spectrum. The maximum number of points
           added per side is equal to max_add_points
        3) Gaussian and Lorentzian widths are generated using the formula:
           min_width < gauss_width + lorentz_width < min(max_width,
                                                          mean(ham_strain),
                                                          spectral_width / spectral_width_factor)
        4) Final interpolation to out_points number of points is performed

        :param min_width: Minimal linewidth in Tesla (T).
            If ham_strain in the magnetic field is greater than min_width, it is ignored
        :param max_width: Maximal linewidth in Tesla (T)
        :param init_interpolation_points: Initial number of points for spectrum interpolation
        :param max_add_points: Maximum number of points to add to each side of the spectrum
        :param out_points: Final number of output points after interpolation
        :param spectral_width_factor: Factor used to determine the maximum spectral width constraint
        :param rng_generator: Optional random number generator
        """
        super().__init__()
        self.min_width = min_width
        self.max_width = max_width
        self.out_points = out_points
        self.max_add_points = max_add_points
        self.init_interpolation_points = init_interpolation_points
        self.spectral_width_factor = spectral_width_factor

        self.post_processor = spectra_manager.PostSpectraProcessing()

        if rng_generator is None:
            self.rng = random.Random(None)
        else:
            self.rng = rng_generator

    def _generate_random_widths(self, mean_ham_strain: torch.Tensor, spectral_width: torch.Tensor):
        """
        Generate random Gauss and Lorentz widths.
        :param mean_ham_strain: Shape [...]
        :return: gauss, lorentz tensors of shape [...]
        """
        batch_shape = mean_ham_strain.shape
        device = mean_ham_strain.device

        max_total = torch.min(self.max_width - mean_ham_strain, spectral_width / self.spectral_width_factor)
        min_total = self.min_width - mean_ham_strain

        max_total = torch.clamp(max_total, min=0.0)
        min_total = torch.clamp(min_total, min=0.0)

        total_width = torch.rand(batch_shape, device=device) * (max_total - min_total) + min_total
        gauss_fraction = torch.rand(batch_shape, device=device)

        gauss = total_width * gauss_fraction
        lorentz = total_width * (1 - gauss_fraction)

        return gauss, lorentz

    def _interpolate_data_after_conv(self, fields: torch.Tensor, spec: torch.Tensor):
        batch_shape = spec.shape[:-1]
        N = spec.shape[-1]
        min_field_pos = fields[..., 0]
        max_field_pos = fields[..., -1]

        spec = torch.nn.functional.interpolate(
            spec.reshape(-1, N).unsqueeze(1),
            size=self.out_points,
            mode='linear',
            align_corners=True
        ).squeeze(1).reshape(*batch_shape, self.out_points)

        steps = torch.linspace(0, 1, self.out_points, device=spec.device, dtype=spec.dtype)
        fields = steps * (max_field_pos - min_field_pos).unsqueeze(-1) + min_field_pos.unsqueeze(-1)
        return fields, spec

    def _init_data_to_covolution(self, min_field_pos: torch.Tensor, max_field_pos: torch.Tensor, spec: torch.Tensor):
        batch_shape = spec.shape[:-1]
        N = spec.shape[-1]
        add_points_right = self.rng.randint(0, self.max_add_points)

        spectral_width = max_field_pos - min_field_pos
        field_step = spectral_width / (N - 1)
        max_points_left = torch.min(min_field_pos / field_step)

        max_points_left = int(max_points_left.item()) - 1
        add_points_left = min(self.rng.randint(0, self.max_add_points), max_points_left)

        target_points = self.init_interpolation_points + add_points_left + add_points_right

        spec_flat = torch.nn.functional.interpolate(
            spec.reshape(-1, N).unsqueeze(1),
            size=self.init_interpolation_points,
            mode='linear',
            align_corners=True
        ).squeeze(1)

        field_step = spectral_width / (self.init_interpolation_points - 1)
        min_field_pos = min_field_pos - field_step * add_points_left
        max_field_pos = max_field_pos + field_step * add_points_right
        new_spec = torch.zeros((math.prod(batch_shape), target_points), device=spec.device, dtype=spec.dtype)

        new_spec[..., add_points_left: target_points - add_points_right] = spec_flat
        new_spec = new_spec.reshape(*batch_shape, target_points)

        steps = torch.linspace(0, 1, target_points, device=spec.device, dtype=spec.dtype)
        fields = steps * (max_field_pos - min_field_pos).unsqueeze(-1) + min_field_pos.unsqueeze(-1)

        return fields, new_spec

    def forward(
            self,
            min_field_pos: torch.Tensor,
            max_field_pos: torch.Tensor,
            spec: torch.Tensor,
            ham_strain: torch.Tensor
    ) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param min_field_pos: Minimum position of the magnetic field. Shape: [...]
        :param max_field_pos: Maximum position of the magnetic field. Shape: [...]
        :param spec: Magnetic resonance spectra. Shape: [..., N], where N is the number of initial points
        :param ham_strain: Hamiltonian strain. Shape: [...]
        :return: Tuple containing:
            - field (torch.Tensor): Output magnetic field in Tesla (T). Shape: [..., out_points]
            - spec (torch.Tensor): Output EPR spectrum in arbitrary units. Shape: [..., out_points]
            - gauss (torch.Tensor): Gaussian linewidth in Tesla (T). Shape: [...]
            - lorentz (torch.Tensor): Lorentzian linewidth in Tesla (T). Shape: [...]
        """
        mean_ham_strain = torch.mean(ham_strain, dim=-1) * constants.PLANCK / constants.BOHR
        spectral_width = max_field_pos - min_field_pos
        gauss, lorentz = self._generate_random_widths(mean_ham_strain, spectral_width)

        field, spec = self._init_data_to_covolution(min_field_pos, max_field_pos, spec)
        spec = self.post_processor(gauss, lorentz, field, spec)
        field, spec = self._interpolate_data_after_conv(field, spec)
        spec = spec / torch.max(abs(spec), dim=-1, keepdim=True)[0]
        return field, spec, gauss, lorentz


class SpectraDistortion(nn.Module):
    def __init__(self,
                 noise_max_level: float = 0.1,
                 baseline_quadratic: float = 0.05,
                 baseline_linear: float = 0.05,
                 baseline_constant: float = 0.05,
                 correct_baseline: bool = True,
                 baseline_points: int = 20
                 ):
        """
        Applies distortions to EPR spectra including baseline drift and noise.

        This module simulates experimental artifacts commonly found in magnetic resonance spectroscopy:
        - Quadratic baseline drift (field-dependent instrumental offset)
        - Gaussian noise with variable amplitude

        The baseline is modeled as a second-order polynomial in normalized field coordinates:
            baseline(x) = a*x² + b*x + c
        where x is the normalized magnetic field position [0, 1].

        :param noise_max_level: Maximum noise amplitude as a fraction of signal.
            Actual noise level is randomly sampled from [0, noise_max_level] for each spectrum

        :param baseline_quadratic: Maximum coefficient for quadratic baseline term (x²).
            Applied to normalized field coordinate

        :param baseline_linear: Maximum coefficient for linear baseline term (x).
            Applied to normalized field coordinate

        :param baseline_constant: Maximum coefficient for constant baseline offset.
            Applied to normalized field coordinate

        :param correct_baseline: If True, applies baseline correction by subtracting the mean
            of edge points from the spectrum
        :param baseline_points: Number of points at each edge of the spectrum used for
            baseline correction (only used if correct_baseline=True)
        """
        super().__init__()
        self.noise_max_level = noise_max_level
        self.baseline_quadratic = baseline_quadratic
        self.baseline_linear = baseline_linear
        self.baseline_constant = baseline_constant
        self.correct_baseline = correct_baseline
        self.baseline_points = baseline_points

    def forward(self, magnetic_field: torch.Tensor, spec: torch.Tensor):
        """
        Apply baseline distortion and noise to EPR spectra.

        The distortion process:
        1) Normalize magnetic field to [0, 1] range
        2) Generate quadratic baseline: a*x² + b*x + c
        3) Add Gaussian noise with random amplitude ∈ [0, noise_max_level]
        4) Optionally correct baseline by subtracting mean of edge points

        :param magnetic_field: Magnetic field positions in Tesla (T). Shape: [..., N]
        :param spec: EPR spectrum intensities. Shape: [..., N]
        :return: Distorted EPR spectrum with same shape as input. Shape: [..., N]

        Note: The noise level is sampled independently for each spectrum in the batch,
              but remains constant across all points within a single spectrum.
        """
        field_min = magnetic_field.min(dim=-1, keepdim=True)[0]
        field_max = magnetic_field.max(dim=-1, keepdim=True)[0]
        field_norm = (magnetic_field - field_min) / (field_max - field_min + 1e-8)

        baseline = (self.baseline_quadratic * field_norm ** 2 +
                    self.baseline_linear * field_norm +
                    self.baseline_constant)

        noise_max_level = torch.rand((spec.shape[:-1]), dtype=spec.dtype, device=spec.device) * self.noise_max_level
        noise = torch.randn_like(spec) * noise_max_level.unsqueeze(-1)
        distorted_spec = spec + baseline + noise
        if self.correct_baseline:
            dims = list(range(1, distorted_spec.dim()))
            baseline = (torch.mean(distorted_spec[..., :self.baseline_points], dim=dims) + torch.mean(
                distorted_spec[..., -self.baseline_points:], dim=dims)) / 2
            distorted_spec = distorted_spec - baseline.unsqueeze(-1)
        return distorted_spec


class SpectraModifier(nn.Module):
    def __init__(self, rng_generator=random.Random(None)):
        """
        Initialize the spectrum modifier pipeline.
        :param rng_generator: Random number generator
        """
        super().__init__()
        self.spec_field_prepare = SpecFieldPrepare(rng_generator=rng_generator)
        self.spec_field_distorter = SpectraDistortion()

    def forward(self,
                min_field_pos: torch.Tensor,
                max_field_pos: torch.Tensor,
                spec: torch.Tensor,
                ham_strain: torch.Tensor):
        """
        Prepare and distort EPR spectra for training.

        Process:
        1. Interpolate spectrum and generate magnetic field grid
        2. Generate Gaussian and Lorentzian linewidth parameters
        3. Apply baseline distortion and noise to simulate experimental artifacts

        :param min_field_pos: Minimum magnetic field position in Tesla (T).
            Shape: [...]
        :param max_field_pos: Maximum magnetic field position in Tesla (T).
            Shape: [...]
        :param spec: Initial magnetic resonance spectrum intensities.
            Shape: [..., num_initial_points]
        :param ham_strain: Hamiltonian strain tensor affecting linewidth constraints measured in Hz.
            Shape: [..., 3]
        :return: Dictionary containing:
            - field (torch.Tensor): Interpolated magnetic field grid in Tesla (T).
                Shape: [..., out_points]
            - spec (torch.Tensor): Clean interpolated spectrum.
                Shape: [..., out_points]
            - spec_distorted (torch.Tensor): Distorted spectrum with baseline and noise.
                Shape: [..., out_points]
            - gauss (torch.Tensor): Generated Gaussian linewidth in Tesla (T).
                Shape: [...]
            - lorentz (torch.Tensor): Generated Lorentzian linewidth in Tesla (T).
                Shape: [...]
        """
        field, spec, gauss, lorentz = self.spec_field_prepare(min_field_pos, max_field_pos, spec, ham_strain)
        spec_distorted = self.spec_field_distorter(field, spec)
        return {
            "field": field,
            "spec": spec,
            "spec_distorted": spec_distorted,
            "gauss": gauss,
            "lorentz": lorentz
        }