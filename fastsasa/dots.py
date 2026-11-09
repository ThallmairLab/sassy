"""
Dots on the unit sphere: geodesic subdivisions of an icosahedron or of a pentakis-dodecahedron.
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
	c72 = np.cos(np.radians(72.0))
	rh = np.sqrt(1.0 - 2.0 * c72) / (1.0 - c72)
	rg = c72 / (1.0 - c72)
	v = [[0, 0, 1]]
	v += [[rh * np.cos(np.radians(a)), rh * np.sin(np.radians(a)), rg] for a in (72, 144, 216, 288, 0)]
	v += [[rh * np.cos(np.radians(a)), rh * np.sin(np.radians(a)), -rg] for a in (36, 108, 180, 252, 324)]
	v += [[0, 0, -1]]
	return np.array(v, float)

def _fill_face(vi: np.ndarray, vj: np.ndarray, vk: np.ndarray, tess: int) -> list[np.ndarray]:
	"""Interior dots of the spherical triangle (vi, vj, vk) at tessellation level tess."""
	tl, tl2 = np.array([(tl, tl2) for tl in range(1, tess - 1) for tl2 in range(1, tess - tl)]).reshape(-1, 2).T
	if not len(tl):
		return []
	pji = _divarc(vj, vi, tl, tess)
	pki = _divarc(vk, vi, tl, tess)
	pij = _divarc(vi, vj, tl2, tess)
	pkj = _divarc(vk, vj, tl2, tess)
	pik = _divarc(vi, vk, tess - tl - tl2, tess)
	pjk = _divarc(vj, vk, tess - tl - tl2, tess)
	p = (_divarc(pki, pji, tl2, tess - tl)
	     + _divarc(pkj, pij, tl, tess - tl2)
	     + _divarc(pjk, pik, tl, tl + tl2))
	return list(p / np.linalg.norm(p, axis=-1, keepdims=True))

def _d2(a: np.ndarray, b: np.ndarray) -> float:
	return np.sum((a - b) ** 2)

def ico_dot_arc(densit: int) -> np.ndarray:
	"""Geodesic icosahedron: 10*tess^2 + 2 dots."""
	tess = max(int(np.ceil(np.sqrt((densit - 2.0) / 10.0))), 1)
	v = _icosahedron()
	pts = list(v)
	if tess > 1:
		a = _d2(v[0], v[1])                       # squared icosahedron edge
		e = lambda i, j: abs(_d2(v[i], v[j]) - a) < 1e-4
		for i in range(11):                       # 30 edges
			for j in range(i + 1, 12):
				if e(i, j):
					pts += list(_divarc(v[i], v[j], np.arange(1, tess), tess))
		for i in range(10):                       # 20 faces
			for j in range(i + 1, 11):
				for k in range(j + 1, 12):
					if e(i, j) and e(i, k) and e(j, k):
						pts += _fill_face(v[i], v[j], v[k], tess)
	return np.array(pts)

def ico_dot_dod(densit: int) -> np.ndarray:
	"""Geodesic pentakis-dodecahedron: 30*tess^2 + 2 dots."""
	tess = max(int(np.ceil(np.sqrt((densit - 2.0) / 30.0))), 1)
	v = _icosahedron()
	a = _d2(v[0], v[1])
	e = lambda i, j: abs(_d2(v[i], v[j]) - a) < 1e-4
	cen = []                                      # 20 face centres = dodecahedron vertices
	for i in range(10):
		for j in range(i + 1, 11):
			for k in range(j + 1, 12):
				if e(i, j) and e(i, k) and e(j, k):
					c = v[i] + v[j] + v[k]
					cen.append(c / np.linalg.norm(c))
	w = np.vstack([v, cen])                       # 32 vertices
	pts = list(w)
	if tess > 1:
		ai_d = min(_d2(w[0], w[j]) for j in range(12, 32))   # ico vertex - face centre
		adod = min(_d2(w[12], w[j]) for j in range(13, 32))  # dodecahedron edge
		def is_e(i: int, j: int) -> bool:
			if i < 12 and j < 12:
				return False
			return abs(_d2(w[i], w[j]) - (ai_d if (i < 12) != (j < 12) else adod)) < 1e-4
		for i in range(31):                       # 90 edges
			for j in range(max(12, i + 1), 32):
				if is_e(i, j):
					pts += list(_divarc(w[i], w[j], np.arange(1, tess), tess))
		for i in range(12):                       # 60 faces (vertex, centre, centre)
			for j in range(12, 31):
				for k in range(j + 1, 32):
					if is_e(i, j) and is_e(i, k) and is_e(j, k):
						pts += _fill_face(w[i], w[j], w[k], tess)
	return np.array(pts)

@lru_cache(maxsize=None)
def unit_dots(ndots: int) -> np.ndarray:
	"""
	At least ndots dots: whichever scheme gives fewer dots >= ndots
	The array is cached and read-only.
	"""
	arc, dod = ico_dot_arc(ndots), ico_dot_dod(ndots)
	unit = arc if len(arc) <= len(dod) else dod
	unit.flags.writeable = False
	return unit
