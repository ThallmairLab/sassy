"""
Dots on the unit sphere: geodesic subdivisions of an icosahedron or of a pentakis-dodecahedron
(as in GROMACS' gmx sasa).
"""
from functools import lru_cache
import numpy as np
from numpy.typing import ArrayLike


def _divarc(p1: ArrayLike, p2: ArrayLike, div1: ArrayLike, div2: float) -> np.ndarray:
	"""
	Point at fraction div1/div2 along the great-circle arc p1 -> p2.
	Works on arrays of points (..., 3) and fractions (...).
	"""
	p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
	axis = np.cross(p1, p2)
	norm = np.linalg.norm(axis, axis=-1, keepdims=True)
	phi = np.arctan2(norm, np.sum(p1 * p2, axis=-1, keepdims=True)) * (np.asarray(div1, float) / div2)[..., None]
	axis = axis / norm
	p1 = p1 / np.linalg.norm(p1, axis=-1, keepdims=True)
	return np.cos(phi) * p1 + np.sin(phi) * np.cross(axis, p1)

def _icosahedron() -> np.ndarray:
	"""(12, 3) vertices of the icosahedron on the unit sphere."""
	c72 = np.cos(np.radians(72.0))
	rh = np.sqrt(1.0 - 2.0 * c72) / (1.0 - c72)
	rg = c72 / (1.0 - c72)
	vertices = [[0, 0, 1]]
	vertices += [[rh * np.cos(np.radians(a)), rh * np.sin(np.radians(a)), rg] for a in (72, 144, 216, 288, 0)]
	vertices += [[rh * np.cos(np.radians(a)), rh * np.sin(np.radians(a)), -rg] for a in (36, 108, 180, 252, 324)]
	vertices += [[0, 0, -1]]
	return np.array(vertices, float)

def _fill_face(vi: np.ndarray, vj: np.ndarray, vk: np.ndarray, tess: int) -> list[np.ndarray]:
	"""Interior dots of the spherical triangle (vi, vj, vk) at tessellation level tess."""
	i, j = np.array([(i, j) for i in range(1, tess - 1) for j in range(1, tess - i)]).reshape(-1, 2).T
	if not len(i):
		return []
	pji = _divarc(vj, vi, i, tess)
	pki = _divarc(vk, vi, i, tess)
	pij = _divarc(vi, vj, j, tess)
	pkj = _divarc(vk, vj, j, tess)
	pik = _divarc(vi, vk, tess - i - j, tess)
	pjk = _divarc(vj, vk, tess - i - j, tess)
	p = (_divarc(pki, pji, j, tess - i)
	     + _divarc(pkj, pij, i, tess - j)
	     + _divarc(pjk, pik, i, i + j))
	return list(p / np.linalg.norm(p, axis=-1, keepdims=True))

def _distance2(a: np.ndarray, b: np.ndarray) -> float:
	return np.sum((a - b) ** 2)

def _ico_dot_arc(n_dots: int) -> np.ndarray:
	"""Geodesic icosahedron with at least ``n_dots`` dots: 10*tess^2 + 2 dots."""
	tess = max(int(np.ceil(np.sqrt(max(n_dots - 2.0, 0.0) / 10.0))), 1)
	ico = _icosahedron()
	dots = list(ico)
	if tess > 1:
		edge2 = _distance2(ico[0], ico[1])          # squared icosahedron edge
		is_edge = lambda i, j: abs(_distance2(ico[i], ico[j]) - edge2) < 1e-4
		for i in range(11):                         # 30 edges
			for j in range(i + 1, 12):
				if is_edge(i, j):
					dots += list(_divarc(ico[i], ico[j], np.arange(1, tess), tess))
		for i in range(10):                         # 20 faces
			for j in range(i + 1, 11):
				for k in range(j + 1, 12):
					if is_edge(i, j) and is_edge(i, k) and is_edge(j, k):
						dots += _fill_face(ico[i], ico[j], ico[k], tess)
	return np.array(dots)

def _ico_dot_dod(n_dots: int) -> np.ndarray:
	"""Geodesic pentakis-dodecahedron with at least ``n_dots`` dots: 30*tess^2 + 2 dots."""
	tess = max(int(np.ceil(np.sqrt(max(n_dots - 2.0, 0.0) / 30.0))), 1)
	ico = _icosahedron()
	edge2 = _distance2(ico[0], ico[1])
	is_ico_edge = lambda i, j: abs(_distance2(ico[i], ico[j]) - edge2) < 1e-4
	centres = []                                    # 20 face centres = dodecahedron vertices
	for i in range(10):
		for j in range(i + 1, 11):
			for k in range(j + 1, 12):
				if is_ico_edge(i, j) and is_ico_edge(i, k) and is_ico_edge(j, k):
					centre = ico[i] + ico[j] + ico[k]
					centres.append(centre / np.linalg.norm(centre))
	vertices = np.vstack([ico, centres])            # 32 vertices: 12 icosahedron, 20 face centres
	dots = list(vertices)
	if tess > 1:
		vertex_centre2 = min(_distance2(vertices[0], vertices[j]) for j in range(12, 32))
		centre_centre2 = min(_distance2(vertices[12], vertices[j]) for j in range(13, 32)) # dodecahedron edge
		def is_edge(i: int, j: int) -> bool:
			if i < 12 and j < 12:
				return False
			expected = vertex_centre2 if (i < 12) != (j < 12) else centre_centre2
			return abs(_distance2(vertices[i], vertices[j]) - expected) < 1e-4
		for i in range(31):                         # 90 edges
			for j in range(max(12, i + 1), 32):
				if is_edge(i, j):
					dots += list(_divarc(vertices[i], vertices[j], np.arange(1, tess), tess))
		for i in range(12):                         # 60 faces (vertex, centre, centre)
			for j in range(12, 31):
				for k in range(j + 1, 32):
					if is_edge(i, j) and is_edge(i, k) and is_edge(j, k):
						dots += _fill_face(vertices[i], vertices[j], vertices[k], tess)
	return np.array(dots)

@lru_cache(maxsize=None)
def unit_dots(ndots: int) -> np.ndarray:
	"""
	At least ``ndots`` dots on the unit sphere: of the two geodesic schemes the one with fewer
	dots >= ndots. The array is cached and read-only.
	"""
	if ndots < 1:
		raise ValueError(f"The number of dots per atom must be >= 1, got {ndots}.")
	arc, dod = _ico_dot_arc(ndots), _ico_dot_dod(ndots)
	unit = arc if len(arc) <= len(dod) else dod
	unit.flags.writeable = False
	return unit
