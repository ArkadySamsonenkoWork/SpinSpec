import math

BOHR = 9.2740100657 * 1e-24  # J * T-1
PLANCK = 6.62607015 * 1e-34  # J⋅Hz-1
NUCLEAR_MAGNETRON = 5.0507837393 * 1e-27  # J * T-1
BOLTZMANN = 1.380649 * 1e-23  # J * K-1
SPEED = 299_792_458  # m / s

field = 10
def get_k():
    g = 2.0
    energy = g * BOHR * field
    freq = energy / PLANCK
    k = freq * 2 * math.pi / SPEED
    wave_l = 0.01 * k
    return wave_l

print(get_k())

