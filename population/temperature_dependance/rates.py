import torch


class RelaxationRateTempDep:
    """Base class for relaxation speed temperature dependencies."""

    def __call__(self, temp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __sub__(self, other):
        return self + (-other)

    def __add__(self, other):
        return CombinedRate([self, other])

    def __eq__(self, other):
        raise NotImplementedError


class ZeroRate(RelaxationRateTempDep):
    def __call__(self, temp: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(temp)


class CombinedRate(RelaxationRateTempDep):
    """Represents a sum of multiple relaxation speed terms."""

    def __init__(self, terms: list[RelaxationRateTempDep]):
        self.terms = terms

    def __call__(self, temp: torch.Tensor) -> torch.Tensor:
        result = None
        for term in self.terms:
            term_result = term(temp)
            if result is None:
                result = term_result
            else:
                result += term_result
        return result

    def __neg__(self):
        return CombinedRate([-term for term in self.terms])

    def __add__(self, other):
        if isinstance(other, CombinedRate):
            return CombinedRate(self.terms + other.terms)
        else:
            return CombinedRate(self.terms + [other])

    def __eq__(self, other):
        return isinstance(other, CombinedRate) and self.terms == other.terms


class ExponentialRate(RelaxationRateTempDep):
    """Exponential dependence: amplitude * exp((temp - zero_temp)/delta_temp)."""

    def __init__(self,
                 amplitude: torch.Tensor,
                 zero_temp: torch.Tensor,
                 delta_temp: torch.Tensor):

        self.amplitude = amplitude
        self.zero_temp = zero_temp
        self.delta_temp = delta_temp

    def __call__(self, temp: torch.Tensor) -> torch.Tensor:
        temp_reshaped = temp.view(-1, *([1] * self.amplitude.dim()))
        exponent = (temp_reshaped - self.zero_temp) / self.delta_temp
        return torch.exp(exponent) * self.amplitude

    def __add__(self, other):
        if (isinstance(other, ExponentialRate) and
                self.zero_temp == other.zero_temp and
                self.delta_temp == other.delta_temp):
            return ExponentialRate(self.amplitude + other.amplitude, self.zero_temp, self.delta_temp)
        else:
            return super().__add__(other)

    def __eq__(self, other):
        return (isinstance(other, ExponentialRate) and
                torch.equal(self.amplitude, other.amplitude) and
                self.zero_temp == other.zero_temp and
                self.delta_temp == other.delta_temp)


class PowerRate(RelaxationRateTempDep):
    """Power low dependance A * T^(n)"""

    def __init__(self, amplitude: torch.Tensor, power: torch.Tensor):
        self.amplitude = amplitude
        self.power = power

    def __call__(self, temp: torch.Tensor) -> torch.Tensor:
        temp_reshaped = temp.view(-1, *([1] * self.amplitude.dim()))
        return self.amplitude * torch.pow(temp_reshaped, self.power)

    def __neg__(self):
        return PowerRate(-self.amplitude, self.power)

    def __add__(self, other):
        if (isinstance(other, PowerRate)) and (self.power == other.power):
            return PowerRate(self.amplitude + other.amplitude, self.power)
        else:
            return super().__add__(other)

    def __eq__(self, other):
        return (isinstance(other, PowerRate) and
                torch.equal(self.amplitude, other.amplitude) and
                torch.equal(self.power, other.power))


class RelaxationRateMatrix:
    def __init__(self, relaxation_speeds: list[list[RelaxationRateTempDep | float]]):
        self.relaxation_speeds = [
            [self._convert_to_rate(entry) for entry in row]
            for row in relaxation_speeds
        ]
    def _convert_to_rate(self, entry):
        if isinstance(entry, RelaxationRateTempDep):
            return entry
        elif isinstance(entry, (int, float)) and entry == 0.0:
            return ZeroRate()
        else:
            raise TypeError(f"Unsupported type in relaxation_speeds: {type(entry)}. "
                            "Only RelaxationRateTempDep or 0.0 are allowed.")

    def __call__(self, temp: torch.Tensor) -> torch.Tensor:
        """
        :param temp: The time-dependant temperature. The shape is [batch_size, time_size]
        :return: Relaxation rates matrix at temp points
        relaxation matrix at specific temperature.
        The output shape is [..., spin dimension, spin dimension]
        """
        result = [[m(temp) for m in row] for row in self.relaxation_speeds]
        result = torch.stack([torch.stack(row, dim=-1) for row in result], dim=-2)
        return result