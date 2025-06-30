import torch

import typing as tp
from dataclasses import dataclass

class Profile:
    """Base class for temperature / power -time dependencies"""
    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def __add__(self, other: "Profile") -> "CombinedProfile":
        return CombinedProfile([self, other])

    def __sub__(self, other: "Profile") -> "CombinedProfile":
        return self + (-other)

    def __neg__(self) -> "NegatedProfile":
        return NegatedProfile(self)


class CombinedProfile(Profile):
    """Represents sum/difference of multiple temperature profiles."""
    def __init__(self, profiles: list[Profile]):
        self.profiles = profiles

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(time)
        for profile in self.profiles:
            result += profile(time)
        return result


class NegatedProfile(Profile):
    """Represents negation of a temperature profile."""
    def __init__(self, profile: Profile):
        self.profile = profile

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        return -self.profile(time)


class ConstantProfile(Profile):
    """Constant profile for power / temperature"""

    def __init__(self, value: float):
        self.value = torch.tensor(value)

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        return self.value.expand_as(time).clone()


class LinearProfile(Profile):
    """Linear profile (temperature / power) change over time in the specific region:
    T(t) = slope* t / delta t + intercept for t in [start_time, end_time]
    T(t) = intercept for t < start_time
    T(t) = slope + intercept for t > end_time
    delta t = end_time - start_time
    """
    def __init__(self, slope: float, intercept: float, start_time: float, end_time: float):
        """
        :param slope: The slope of the linear curve. The unit is value
        :param intercept: The intercept in K
        :param start_time: time in s
        :param end_time: time in s
        """
        self.slope = torch.tensor(slope)
        self.intercept = torch.tensor(intercept)

        self.start_time = torch.tensor(start_time)
        self.end_time = torch.tensor(end_time)

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        """
        :param time: time in s units. The shape is [...]
        :return: The dependence of the value on time
        """
        final_value = self.slope + self.intercept
        result = torch.zeros_like(time)

        mask_before = time < self.start_time
        result[mask_before] = self.intercept

        delta = self.end_time - self.start_time
        mask_between = (time >= self.start_time) & (time <= self.end_time)
        result[mask_between] = self.slope * ((time[mask_between] - self.start_time) / delta) + self.intercept

        mask_after = time > self.end_time
        result[mask_after] = final_value

        return result


class ExponentialDecayProfile(Profile):
    """Exponential decay to steady state: A(t) = A0 + (A_init - A0)*exp(-t/tau)."""

    def __init__(self, initial_value: float, steady_value: float, decay_tau: float):
        """
        :param initial_value: initial temperature of the system in K
        :param steady_value: final temperature of the system in K
        :param decay_tau: the time of temperature change in s
        """
        self.A_init = torch.tensor(initial_value)
        self.A0 = torch.tensor(steady_value)
        self.decay_tau = torch.tensor(decay_tau)

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        return self.A0 + (self.A_init - self.A0) * torch.exp(-time / self.decay_tau)


class StepProfile(Profile):
    """Instantaneous step at specified time."""

    def __init__(self, step_time: float, initial_value: float, final_temp: float):
        self.step_time = torch.tensor(step_time)
        self.initial_value = torch.tensor(initial_value)
        self.final_value = torch.tensor(final_temp)

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        return torch.where(time < self.step_time, self.initial_value, self.final_value)


class LinearExpDecayProfile(Profile):
    """
    Linear ramp combined with exponential decay.
    T(t) = base + ramp(t) + (initial_offset - base - end_ramp) * exp(-(t - start_time)/tau) for t >= start_time
    Before start: returns intercept
    After end of ramp: ramp is complete, then decay applies to offset

    """
    def __init__(
        self,
        slope: float,
        intercept: float,
        start_time: float,
        end_time: float,
        decay_tau: float,
    ):
        self.slope = torch.tensor(slope)
        self.intercept = torch.tensor(intercept)
        self.start_time = torch.tensor(start_time)
        self.end_time = torch.tensor(end_time)
        self.decay_tau = torch.tensor(decay_tau)

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        """
        :param time: time in s units. The shape is [...]
        :return: The dependence of the value on time
        """
        result = torch.zeros_like(time)

        mask_before = time < self.start_time
        result[mask_before] = self.intercept

        delta = self.end_time - self.start_time
        mask_between = (time >= self.start_time) & (time <= self.end_time)

        time_between = time[mask_between]
        decay_term = torch.exp(-time_between / self.decay_tau)
        result[mask_between] = decay_term * self.slope * ((time_between - self.start_time) / delta) + self.intercept

        mask_after = time > self.end_time
        decay_term = torch.exp(-time[mask_after] / self.decay_tau)
        result[mask_after] = decay_term * self.slope + self.intercept

        return result


@dataclass
class Segment:
    profile: Profile
    start: float
    end: float


class PiecewiseProfile(Profile):
    """Piecewise profile with different dependencies in time segments"""

    def __init__(self, segments: list[Segment]):
        """
        :param segments: List of segment definitions. Each segment should contain:
            - 'profile': string specifying dependency type (linear, exponential, etc.)
            - 'start': start time of the segment
            - 'end': end time of the segment
        """
        super().__init__()
        self.segments = self._validate_segments(segments)

    def _validate_segments(self, segments: list[Segment]):
        prev_end = -torch.inf
        for seg in segments:
            if seg.start < prev_end:
                raise ValueError("Segments must be ordered chronologically without overlaps")

        return segments

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(time)

        for seg in self.segments:
            mask = (time >= seg.start) & (time < seg.end)
            segment_time = time[mask]  # Time relative to segment start
            result[mask] = seg.profile(segment_time)

        return result

    def add_segment(self, segment: Segment):
        """Add a new segment to the piecewise profile"""
        self.segments = self._validate_segments(self.segments + [segment])


class ThermalProperty:
    """Base class for temperature-dependent thermal properties"""

    def __call__(self, temperature: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ConstantProperty(ThermalProperty):
    """Constant value independent of temperature"""
    def __init__(self, value: float):
        self.value = torch.tensor(value)

    def __call__(self, _: torch.Tensor) -> torch.Tensor:
        return self.value


class DebyeHeatCapacity(ThermalProperty):
    """Debye model heat capacity with temperature dependence"""

    def __init__(self, debye_temp: float, molar_mass: float, n_atoms: int):
        self.debye_temp = torch.tensor(debye_temp)
        self.molar_mass = torch.tensor(molar_mass)
        self.n_atoms = torch.tensor(n_atoms)
        self.R = torch.tensor(8.314)

    def __call__(self, temperature: torch.Tensor) -> torch.Tensor:
        x = self.debye_temp / temperature.clamp(min=1e-3)  # Avoid division by zero
        integral = (x ** 3 * torch.exp(-x)) / (1 - torch.exp(-x)) ** 2
        return 9 * self.R * self.n_atoms * integral * self.molar_mass


class ThermalEquationProfile(Profile):
    """Temperature profile from power input and thermal properties"""

    def __init__(self, power_profile: Profile,
                 heat_capacity: tp.Callable[[torch.Tensor], torch.Tensor],
                 initial_temp: float = 3.5):
        super().__init__("temperature")
        self.power_profile = power_profile
        self.heat_capacity = heat_capacity
        self.initial_temp = torch.tensor(initial_temp)

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        dt = time[1] - time[0] if len(time) > 1 else 1.0
        T = torch.zeros_like(time)
        T[0] = self.initial_temp

        for i in range(1, len(time)):
            P = self.power_profile(time[:i])
            C_th = self.heat_capacity(T[i - 1])

            # Solve dT/dt = P(t) * R_th(T) / C_th(T)
            dTdt = P[-1] / C_th
            T[i] = T[i - 1] + dTdt * dt
        return T