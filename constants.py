BOHR = 9.2740100657 * 1e-24  # J * T-1
PLANCK = 6.62607015 * 1e-34  # J⋅Hz-1
NUCLEAR_MAGNETRON = 5.0507837393 * 1e-27  # J * T-1
BOLTZMANN = 1.380649 * 1e-23  # J * K-1


def unit_converter(values, conversion=""):
    if conversion == "Hz_to_K":
        return values * PLANCK / BOLTZMANN

