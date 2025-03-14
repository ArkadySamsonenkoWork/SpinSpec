from dataclasses import dataclass, field
import os
import pickle
import torch
import math
from torch import nn

import constants
import utils

@dataclass(frozen=True)
class SpinMatricesHalf:
    """Spin matrices for spin-1/2 particles."""
    x: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 0.5], [0.5, 0]], dtype=torch.complex64))
    y: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, -0.5j], [0.5j, 0]], dtype=torch.complex64))
    z: torch.Tensor = field(default_factory=lambda: torch.tensor([[0.5, 0], [0, -0.5]], dtype=torch.complex64))
    plus: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64))
    minus: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64))

    @property
    def matrices(self):
        return [self.x, self.y, self.z]

@dataclass(frozen=True)
class SpinMatricesOne:
    """Spin matrices for spin-1/2 particles."""
    x: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 0.5], [0.5, 0]], dtype=torch.complex64))
    y: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, -0.5j], [0.5j, 0]], dtype=torch.complex64))
    z: torch.Tensor = field(default_factory=lambda: torch.tensor([[0.5, 0], [0, -0.5]], dtype=torch.complex64))
    plus: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64))
    minus: torch.Tensor = field(default_factory=lambda: torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64))

    @property
    def matrices(self):
        return [self.x, self.y, self.z]

# Лучше этим пользоваться
def get_spin_operators(s):
    """Generate spin matrices for a given spin s."""
    s = float(s)
    dim = int(2 * s + 1)
    if not (2 * s).is_integer():
        raise ValueError("Spin must be an integer or half-integer.")

    sz = torch.diag(torch.tensor([s - i for i in range(dim)], dtype=torch.complex64))
    splus = torch.zeros((dim, dim), dtype=torch.complex64)
    sminus = torch.zeros((dim, dim), dtype=torch.complex64)

    for i in range(dim):
        m_i = s - i
        if m_i + 1 <= s:
            j = i - 1
            value = math.sqrt((s - m_i) * (s + m_i + 1))
            splus[j, i] = value
        if m_i - 1 >= -s:
            j = i + 1
            value = math.sqrt((s + m_i) * (s - m_i + 1))
            sminus[j, i] = value

    # Sx and Sy from ladder operators
    sx = (splus + sminus) / 2
    sy = (splus - sminus) / (2j)
    return {
        'x': sx,
        'y': sy,
        'z': sz,
        'plus': splus,
        'minus': sminus,
        'matrices': (sx, sy, sz)
    }


@dataclass
class Particle:
    """Represents a particle with spin and associated matrices.
    Spin must be an integer or half-integer.
    """
    spin: float
    spin_matrices: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = field(init=False)
    identity: torch.Tensor = field(init=False)

    def __post_init__(self):
        dim = int(2 * self.spin + 1)
        self.identity = torch.eye(dim, dtype=torch.complex64)
        self.spin_matrices = get_spin_operators(self.spin)["matrices"]


@dataclass
class Electron(Particle):
    """Represents the electron Particle
    Spin must be an integer or half-integer.
    """


class Nucleus(Particle):
    """Represents a nucleus with spin and g-factor loaded from a pre-parsed database."""
    _isotope_data = None
    _data_loaded = False  # To load data only one time

    def __init__(self, nucleus_str: str):
        self.nucleus_str = nucleus_str
        if not Nucleus._data_loaded:
            data_path = self._get_data_path("nuclei_db/nuclear_data.pkl")
            Nucleus._load_isotope_data(data_path)
        spin, g_factor = self._parse_nucleus_str(nucleus_str)
        super().__init__(spin)
        self.g_factor = torch.tensor(g_factor)

    @classmethod
    def _load_isotope_data(cls, data_path: str):
        """Load isotope data from a pickle file."""
        try:
            with open(data_path, 'rb') as f:
                cls._isotope_data = pickle.load(f)
            cls._data_loaded = True
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Isotope data file '{data_path}' not found.")

    def _get_data_path(self, filename: str) -> str:
        """Get the absolute path to the data file, relative to the location of this class."""
        class_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(class_dir, filename)

    def _parse_nucleus_str(self, nucleus_str: str) -> tuple[float, float]:
        """Extract nucleons and symbol from the nucleus string (e.g., '14N' -> (14, 'N'))."""
        data = Nucleus._isotope_data.get(nucleus_str)
        if not data:
            raise KeyError(f"No data found for nucleus: {self.nucleus_str}")
        return (data['spin'], data['gn'])