import torch
import numpy as np

BOHR = 9.2740100657 * 1e-24  # J * T-1
PLANCK = 6.62607015 * 1e-34  # J⋅Hz-1
NUCLEAR_MAGNETRON = 5.0507837393 * 1e-27  # J * T-1
BOLTZMANN = 1.380649 * 1e-23  # J * K-1
E_CHARGE = 1.602176634e-19  # C (for eV ↔ J)
SPEED = 299_792_458  # m / s

_CONVERTERS = {
    # energy ↔ frequency
    "Hz_to_J": lambda v: v * PLANCK,
    "J_to_Hz": lambda v: v / PLANCK,

    # energy ↔ temperature
    "J_to_K": lambda v: v / BOLTZMANN,
    "K_to_J": lambda v: v * BOLTZMANN,

    # frequency ↔ temperature (via E = h·ν, E = k·T)
    "Hz_to_K": lambda v: v * PLANCK / BOLTZMANN,
    "K_to_Hz": lambda v: v * BOLTZMANN / PLANCK,

    # energy ↔ magnetic field strength
    "Hz_to_T_e": lambda v: v * PLANCK / BOHR,
    "T_to_Hz_e": lambda v: v * BOHR / PLANCK,
    "Hz_to_T_n": lambda v: v * PLANCK / NUCLEAR_MAGNETRON,
    "T_to_Hz_n": lambda v: v * NUCLEAR_MAGNETRON / PLANCK,

    # J ↔ eV
    "J_to_eV": lambda v: v / E_CHARGE,
    "eV_to_J": lambda v: v * E_CHARGE,

    # cm-1 ↔ Hz
    "cm-1_to_Hz": lambda v: v * SPEED * 100,
    "Hz_to_cm-1": lambda v: v / (SPEED * 100),

    # cm-1 ↔ Hz
    "cm-1_to_K": lambda v: v * PLANCK * SPEED * 100 / BOLTZMANN,
    "K_to_cm-1": lambda v: v * BOLTZMANN / (PLANCK * SPEED * 100),

    # cm-1 ↔ T
    "cm-1_to_T_e": lambda v: v * PLANCK * SPEED * 100 / BOHR,
    "T_to_cm-1_e": lambda v: v * BOHR / (PLANCK * SPEED * 100),
    "cm-1_to_T_n": lambda v: v * PLANCK * SPEED * 100 / NUCLEAR_MAGNETRON,
    "T_to_cm-1_n": lambda v: v * NUCLEAR_MAGNETRON / (PLANCK * SPEED * 100),
}


def unit_converter(values: torch.Tensor | np.ndarray | float, conversion: str):
    """
    Convert `values` (scalar or torch.Tensor on CPU or GPU)
    according to the given conversion key.


    :param values:  torch.Tensor, np.ndarray, or float
        The numerical value(s) to be converted. Can be a scalar, numpy array,
        or PyTorch tensor (CPU or GPU).

    :param conversion: str
        The conversion type specified as "from_to" format. Supported conversions:
        Energy ↔ Frequency:
        - "Hz_to_J": frequency (Hz) → energy (Joules)
        - "J_to_Hz": energy (Joules) → frequency (Hz)

        Energy ↔ Temperature:
        - "J_to_K": energy (Joules) → temperature (Kelvin)
        - "K_to_J": temperature (Kelvin) → energy (Joules)

        Frequency ↔ Temperature:
        - "Hz_to_K": frequency (Hz) → temperature (Kelvin)
        - "K_to_Hz": temperature (Kelvin) → frequency (Hz)

        Energy/Frequency ↔ Magnetic Field:
        - "Hz_to_T_e": frequency (Hz) → magnetic field (Tesla) for electrons
        - "T_to_Hz_e": magnetic field (Tesla) → frequency (Hz) for electrons
        - "Hz_to_T_n": frequency (Hz) → magnetic field (Tesla) for nuclei
        - "T_to_Hz_n": magnetic field (Tesla) → frequency (Hz) for nuclei

        Energy Units:
        - "J_to_eV": energy (Joules) → energy (electron volts)
        - "eV_to_J": energy (electron volts) → energy (Joules)

        Wavenumber Conversions:
        - "cm-1_to_Hz": wavenumber (cm⁻¹) → frequency (Hz)
        - "Hz_to_cm-1": frequency (Hz) → wavenumber (cm⁻¹)
        - "cm-1_to_K": wavenumber (cm⁻¹) → temperature (Kelvin)
        - "K_to_cm-1": temperature (Kelvin) → wavenumber (cm⁻¹)
        - "cm-1_to_T_e": wavenumber (cm⁻¹) → magnetic field (Tesla) for electrons
        - "T_to_cm-1_e": magnetic field (Tesla) → wavenumber (cm⁻¹) for electrons
        - "cm-1_to_T_n": wavenumber (cm⁻¹) → magnetic field (Tesla) for nuclei
        - "T_to_cm-1_n": magnetic field (Tesla) → wavenumber (cm⁻¹) for nuclei
    """
    try:
        fn = _CONVERTERS[conversion]
    except KeyError:
        valid = ", ".join(_CONVERTERS.keys())
        raise ValueError(f"Unknown conversion ‘{conversion}’. "
                         f"Supported: {valid}")
    return fn(values)


