"""
Environment of a SASA: atoms (e.g. a protein) that cover the surface of the bodies
(e.g. a bound ligand) without being counted themselves.

For every body, the environment atoms within reach are copied next to it, in its
periodic image, so that each body is covered by the environment but not by the
other bodies.
"""
import numpy as np
from numpy.typing import ArrayLike
from MDAnalysis.lib.distances import capped_distance, minimize_vectors

from .units import to_nm


def box_nm(dimensions: ArrayLike | None) -> np.ndarray | None:
	"""
	MDAnalysis box dimensions ([a, b, c] in Å, [alpha, beta, gamma] in degrees) with the
	lengths in nm, or None without a (valid) box.
	"""
	if dimensions is None:
		return None
	box = np.asarray(dimensions, dtype=np.float64).copy()
	if box.shape != (6,) or not np.all(box[:3] > 0):
		return None
	box[:3] = to_nm(box[:3])
	return box


def environment_copies(xyz: ArrayLike, body: ArrayLike, env_xyz: ArrayLike, cutoff: float,
                       box: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	The environment atoms within ``cutoff`` of every body, each in the periodic image of
	the body (minimum image to a body atom within reach, so bodies are assumed whole).

	Parameters
	----------
	xyz : np.ndarray
		(n, 3) positions (nm) of the atoms of the bodies
	body : np.ndarray
		(n,) body index of every atom
	env_xyz : np.ndarray
		(m, 3) positions (nm) of the environment atoms
	cutoff : float
		distance (nm) up to which an environment atom can bury a body atom (the sum of
		their largest expanded radii)
	box : np.ndarray, optional
		box dimensions with lengths in nm (see :func:`box_nm`); None: no periodic images

	Returns
	-------
	bodies : np.ndarray
		(k,) body of every copy
	env : np.ndarray
		(k,) index in ``env_xyz`` of every copy; an environment atom near several bodies
		is copied once per body
	positions : np.ndarray
		(k, 3) positions (nm) of the copies, in the periodic image of their body
	"""
	xyz = np.asarray(xyz, dtype=np.float64)
	env_xyz = np.asarray(env_xyz, dtype=np.float64)
	body = np.asarray(body, dtype=np.int64)
	empty = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, 3))
	if not len(xyz) or not len(env_xyz):
		return empty
	pairs = np.asarray(capped_distance(xyz, env_xyz, max_cutoff=cutoff, box=box, return_distances=False),
	                   dtype=np.int64).reshape(-1, 2)
	if not len(pairs):
		return empty
	# one copy per (body, environment atom), anchored to the first body atom within reach
	keys = body[pairs[:, 0]] * len(env_xyz) + pairs[:, 1]
	_, first = np.unique(keys, return_index=True)
	i, j = pairs[first, 0], pairs[first, 1]
	positions = env_xyz[j]
	if box is not None:
		positions = xyz[i] + minimize_vectors(positions - xyz[i], box)
	return body[i], j, positions
