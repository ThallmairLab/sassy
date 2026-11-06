"""
Shrake-Rupley burial test.

A dot u (unit vector) of sphere i with expanded radius R_i lies inside the
expanded sphere of a neighbour j at d_ij = x_j - x_i exactly if

	u . d_ij > (R_i^2 + |d_ij|^2 - R_j^2) / (2 R_i),

which is tested in a compiled (numba) loop over all dots, starting with the
neighbour that buried the previous dot. Neighbour pairs are found with a k-d
tree; spheres only bury dots of spheres in the same block (e.g. the same
molecule in the same frame).
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from numba import njit, prange

from .dots import unit_dots
from .radii import PROBE_RADIUS


@njit(parallel=True, cache=True)
def _test_dots(starts: np.ndarray, vec: np.ndarray, threshold: np.ndarray, unit: np.ndarray,
               exposed: np.ndarray) -> None:
	"""
	Mark the dots of every sphere that are not inside any neighbouring sphere.

	The neighbours of sphere a are ``starts[a]:starts[a + 1]``, closest first;
	dot u of sphere a is inside neighbour m if ``u . vec[m] > threshold[m]``.
	The neighbour that buried the previous dot is tested first.
	"""
	for a in prange(len(starts) - 1):
		first, end = starts[a], starts[a + 1]
		last = first
		for k in range(unit.shape[0]):
			if first == end:
				exposed[a, k] = True
				continue
			ux, uy, uz = unit[k, 0], unit[k, 1], unit[k, 2]
			if ux * vec[last, 0] + uy * vec[last, 1] + uz * vec[last, 2] > threshold[last]:
				exposed[a, k] = False
				continue
			free = True
			for m in range(first, end):
				if ux * vec[m, 0] + uy * vec[m, 1] + uz * vec[m, 2] > threshold[m]:
					last = m
					free = False
					break
			exposed[a, k] = free

@njit(parallel=True, cache=True)
def _neighbour_lists(xyz: np.ndarray, ext: np.ndarray, block: np.ndarray,
                     pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Neighbour lists of the spheres from candidate ``pairs`` (i < j):
		- only pairs with overlapping spheres in the same block
		- every pair in both directions
		- the neighbours of every sphere sorted closest first (bury the most dots)

	Returns
	-------
	starts : np.ndarray
		(n + 1,) the neighbours of sphere a are ``starts[a]:starts[a + 1]``
		reference array for ``vec`` and ``threshold``
	vec : np.ndarray
		(n_neighbours, 3) vector from the sphere to its neighbour
	threshold : np.ndarray
		(n_neighbours,) dot u of the sphere is inside the neighbour if ``u . vec > threshold``
	"""
	n = len(xyz)
	keep = np.zeros(len(pairs), dtype=np.bool_)
	pair_dist2 = np.empty(len(pairs))
	starts = np.zeros(n + 1, dtype=np.int64)
	for p in range(len(pairs)):
		i, j = pairs[p, 0], pairs[p, 1]
		if block[i] != block[j]:
			continue
		d2 = 0.0
		for c in range(3):
			d2 += (xyz[j, c] - xyz[i, c])**2
		if d2 < (ext[i] + ext[j])**2:
			keep[p] = True
			pair_dist2[p] = d2
			starts[i + 1] += 1
			starts[j + 1] += 1
	for a in range(n):
		starts[a + 1] += starts[a]
	fill = starts[:-1].copy()
	vec = np.empty((starts[n], 3))
	dist2 = np.empty(starts[n])
	threshold = np.empty(starts[n])
	for p in range(len(pairs)):
		if not keep[p]:
			continue
		i, j = pairs[p, 0], pairs[p, 1]
		d2 = pair_dist2[p]
		for own, other, sign in ((i, j, 1.0), (j, i, -1.0)):
			m = fill[own]
			fill[own] += 1
			for c in range(3):
				vec[m, c] = sign * (xyz[j, c] - xyz[i, c])
			dist2[m] = d2
			threshold[m] = (ext[own]**2 + d2 - ext[other]**2) / (2 * ext[own])
	# insertion sort of every (short) neighbour list by distance
	for a in prange(n):
		for m in range(starts[a] + 1, starts[a + 1]):
			k = m
			while k > starts[a] and dist2[k - 1] > dist2[k]:
				dist2[k - 1], dist2[k] = dist2[k], dist2[k - 1]
				threshold[k - 1], threshold[k] = threshold[k], threshold[k - 1]
				for c in range(3):
					vec[k - 1, c], vec[k, c] = vec[k, c], vec[k - 1, c]
				k -= 1
	return starts, vec, threshold

def _exposed_dots(xyz: ArrayLike, ext: ArrayLike, block: ArrayLike, unit: np.ndarray) -> np.ndarray:
	"""
	Find the exposed dots of a set of spheres.

	Parameters
	----------
	xyz : np.ndarray
		(n, 3) sphere centres
	ext : np.ndarray
		(n,) radii of the spheres (van der Waals + probe radius), all > 0
	block : np.ndarray
		(n,) block of every sphere: spheres only bury dots of spheres in the same block
		(used for to seperate molecules or frames in a trajectory)
	unit : np.ndarray
		(n_dots, 3) unit-sphere dots

	Returns
	-------
	np.ndarray
		(n, n_dots) bool mask, True for the dots that are not inside another sphere
	"""
	xyz = np.ascontiguousarray(xyz, dtype=np.float64)
	ext = np.ascontiguousarray(ext, dtype=np.float64)
	block = np.ascontiguousarray(block, dtype=np.int64)
	if len(xyz) > 1:
		pairs = cKDTree(xyz).query_pairs(2 * ext.max(), output_type='ndarray').astype(np.int64)
	else:
		pairs = np.zeros((0, 2), dtype=np.int64)
	starts, vec, threshold = _neighbour_lists(xyz, ext, block, pairs)
	exposed = np.empty((len(xyz), len(unit)), dtype=bool)
	_test_dots(starts, vec, threshold, np.ascontiguousarray(unit, dtype=np.float64), exposed)
	return exposed

def surface_dots(xyz: ArrayLike, radii: ArrayLike, probe: float = PROBE_RADIUS,
                 ndots: int = 240) -> tuple[np.ndarray, np.ndarray, float]:
	"""
	Generate dots of the solvent accessible surface of a set of spheres.

	Parameters
	----------
	xyz : np.ndarray
		(n_atoms, 3) coordinates of the atoms
	radii : np.ndarray
		(n_atoms,) van der Waals radii of the atoms, same unit as ``xyz``. 
		Particles with radius 0 (e.g. dummy beads) are ignored in the calculation.
	probe : float
		probe radius, in the units of ``xyz`` (default in nm)
	ndots : int
		minimum number of dots per sphere

	Returns
	-------
	dots : np.ndarray
		(n_dots, 3) dot positions
	owners : np.ndarray
		(n_dots,) index of the particle each dot belongs to
	area : float
		solvent accessible surface area estimated from the dots (units of ``xyz`` squared)
	"""
	xyz = np.asarray(xyz, dtype=float)
	radii = np.asarray(radii, dtype=float)
	unit = unit_dots(int(ndots)) # generate unit sphere dots as template for all spheres
	active = np.flatnonzero(radii > 0) # remove dummy beads (radius=0)
	centres = xyz[active]
	expanded = radii[active] + probe
	exposed = _exposed_dots(centres, expanded, np.zeros(len(active), dtype=int), unit)
	area = float(np.sum(exposed.mean(axis=1) * 4 * np.pi * expanded**2))
	dots = (centres[:, None, :] + expanded[:, None, None] * unit[None])[exposed] # coordinates of dots
	owners = active[np.nonzero(exposed)[0]] # index of the particle each dot belongs to
	return dots, owners, area
