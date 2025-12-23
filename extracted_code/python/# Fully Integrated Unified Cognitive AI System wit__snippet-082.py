# from .core_math import Vector, CustomRandom

from .e8_lattice import E8Lattice

class PerspectiveEngine:
def __init__(self, e8_lattice: E8Lattice, rng: CustomRandom, k_mirror=0.5):
