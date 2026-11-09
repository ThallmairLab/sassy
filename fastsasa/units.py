"""
Length units. MDAnalysis gives and takes coordinates in Å; fastsasa works in nm
throughout. Coordinates are converted with :func:`to_nm` as soon as they are read
from MDAnalysis, and back with :func:`to_angstrom` only where they are handed to
MDAnalysis again (e.g. for writing a file).
"""
import numpy as np
from numpy.typing import ArrayLike

NM_PER_ANGSTROM = 0.1
ANGSTROM_PER_NM = 10.0 # multiplied, not divided by 0.1: 0.3 nm -> 3.0, not 2.9999999999999996 Å


def to_nm(angstrom: ArrayLike) -> np.ndarray:
	"""Coordinates in Å (e.g. from MDAnalysis) -> float64 array in nm."""
	return np.asarray(angstrom, dtype=np.float64) * NM_PER_ANGSTROM


def to_angstrom(nm: ArrayLike) -> np.ndarray:
	"""Coordinates in nm -> float64 array in Å (e.g. for MDAnalysis)."""
	return np.asarray(nm, dtype=np.float64) * ANGSTROM_PER_NM
