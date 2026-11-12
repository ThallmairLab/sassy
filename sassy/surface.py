"""
Dot representation of the solvent accessible surface (Connolly-style dot
surface), for visual inspection and for comparing
atomistic, mapped and coarse-grained structures.
"""
from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings
import numpy as np
from numpy.typing import ArrayLike
import MDAnalysis as mda
from MDAnalysis.analysis.align import rotation_matrix
from scipy.stats import gaussian_kde

from .dots import unit_dots
from .engine import _surface_dots, warn_coincident
from .groups import atom_label
from .radii import PROBE_RADIUS, atom_radii, radii_remarks
from .units import to_angstrom, to_nm

logger = logging.getLogger(__name__)

SURFACE_DOTS = 240 # default minimum number of dots per atom of a written surface (-> 252 dots)


@dataclass
class Surface:
	"""
	Solvent accessible surface dots of a structure, see :func:`generate_surface` and
	:meth:`SASA.write_surface`.

	Attributes
	----------
	atoms : MDAnalysis.AtomGroup
		the atoms of the structure
	positions : np.ndarray
		(n_atoms, 3) positions (nm) of the atoms
	dots : np.ndarray
		(n_dots, 3) positions (nm) of the exposed surface dots
	owners : np.ndarray
		(n_dots,) index in ``atoms`` of the atom every dot belongs to
	area : float
		SASA (nm^2) estimated from the dots
	frame : int
		trajectory frame of the structure
	time : float
		time (ps) of the frame
	rmsd : float or None
		RMSD (nm) to the reference structure the frame was matched to and superposed
		onto, None without a reference
	remarks : list of str
		REMARK records of the written PDB file
	molecule : int or None
		index of the molecule in the :class:`SASA` (from :meth:`SASA.write_surface`), else None
	"""
	atoms: mda.AtomGroup
	positions: np.ndarray
	dots: np.ndarray
	owners: np.ndarray
	area: float
	frame: int
	time: float
	rmsd: float | None = None
	remarks: list[str] | None = None
	molecule: int | None = None

	def write(self, fname: str | Path) -> None:
		"""
		Write the structure and its dots to a PDB file: the atoms first as ATOM records
		(chain A), then the dots as HETATM records (chain X) with atom name DOT and the
		residue name and id of the atom they belong to.
		"""
		write_surface_pdb(fname, self)


def generate_surface(atoms: mda.AtomGroup, radii: ArrayLike | None = None, probe: float = PROBE_RADIUS,
                     block: ArrayLike | None = None, frame: int | None = None,
                     reference: ArrayLike | mda.AtomGroup | None = None, frames: ArrayLike | None = None,
                     ndots: int = SURFACE_DOTS, cg: bool = False, custom_radii: bool | None = None) -> Surface:
	"""
	Solvent accessible surface dots (Connolly-style dot surface) of ``atoms`` in one
	frame of their trajectory.

	Parameters
	----------
	atoms : MDAnalysis.AtomGroup
		atoms of the structure
	radii : np.ndarray, optional
		(n_atoms,) van der Waals radii (nm); default :func:`atom_radii` of ``atoms``
	probe : float
		probe radius (nm)
	block : np.ndarray, optional
		(n_atoms,) molecule index of every atom: atoms only bury dots of atoms with the
		same index (molecules on their own); default all atoms bury each other
	frame : int, optional
		trajectory frame (negative counts from the end); default the current frame
	reference : np.ndarray or MDAnalysis.AtomGroup, optional
		reference structure, (n_atoms, 3) positions (nm) or an AtomGroup with as many
		atoms (in the same order, e.g. a mapped atomistic structure for a CG model):
		instead of ``frame``, the frame of ``frames`` with the lowest RMSD to it is used,
		superposed onto it, so that both surfaces share one coordinate frame
	frames : array-like of int, optional
		frames searched for the ``reference``; default all frames
	ndots : int
		minimum number of dots per atom
	cg : bool
		Martini CG radii (default radii, and the REMARK records)
	custom_radii : bool, optional
		whether ``radii`` are custom (not the defaults), for the REMARK records; default
		True if ``radii`` are given

	Returns
	-------
	Surface
		structure, dots and area (nm^2, from the dots); write it with :meth:`Surface.write`
	"""
	if frame is not None and reference is not None:
		raise ValueError("Give either a frame or a reference structure, not both.")
	custom = radii is not None if custom_radii is None else custom_radii
	radii = atom_radii(atoms, cg=cg) if radii is None else np.asarray(radii, dtype=float)
	trajectory = atoms.universe.trajectory
	current = trajectory.ts.frame

	# select the frame and get the positions of the atoms
	rmsd = None
	try:
		if reference is not None: # select structure closest to the reference structure and superpose it onto it
			reference = to_nm(reference.positions) if hasattr(reference, 'positions') else np.asarray(reference, dtype=float)
			if reference.shape != (atoms.n_atoms, 3):
				raise ValueError(f"The reference structure has {len(reference)} atoms, but the surface has "
				                 f"{atoms.n_atoms}; select one molecule with a matching reference.")
			frames = np.arange(trajectory.n_frames) if frames is None else np.asarray(frames)
			coordinates = np.array([to_nm(trajectory[f].positions[atoms.indices]) for f in frames])
			best, positions, rmsd = best_matching_frame(coordinates, reference)
			frame = int(frames[best])
			logger.info("frame %d is closest to the reference structure (RMSD %.3f nm), superposed onto it",
			            frame, rmsd)
		else: # select the given frame or fall back to current frame 
			frame = current if frame is None else int(np.arange(trajectory.n_frames)[frame]) # IndexError
			positions = to_nm(trajectory[frame].positions[atoms.indices])
		time = float(trajectory[frame].time)
	finally:
		trajectory[current] # leave the trajectory on the frame it was on

	# generate dots
	dots, owners, area, coincident = _surface_dots(positions, radii, probe=probe, ndots=ndots, block=block)
	warn_coincident(coincident, label=lambda i: f"{atom_label(atoms[i])} (frame {frame})")

	# write remarks for the PDB file
	n_dots = len(unit_dots(int(ndots)))
	remarks = [f"Structure (ATOM) and solvent accessible surface dots (HETATM), sassy {_version()}",
	           f"Frame {frame}, time {time:g} ps, SASA {area:.3f} nm^2, {n_dots} dots per atom"]
	if rmsd is not None:
		remarks.append(f"Superposed onto the reference structure, RMSD {rmsd:.3f} nm")
	remarks += radii_remarks(atoms, cg=cg, custom=custom, probe=probe)
	logger.debug("surface of frame %d: %d atoms, %d dots, SASA %.3f nm^2", frame, atoms.n_atoms, len(dots), area)
	return Surface(atoms, positions, dots, owners, area, frame, time, rmsd, remarks)


def _version() -> str:
	from . import __version__ # not at import time: the package imports this module
	return __version__


def mode(values: ArrayLike, n_grid: int = 1024) -> float:
	"""
	Most probable value: the maximum of a Gaussian kernel density estimate of the values
	(Scott's bandwidth), on a grid of ``n_grid`` points from their minimum to maximum.
	For bimodal distributions this is the higher peak, where mean and median may fall
	between the peaks. With fewer than two distinct values, the median.
	"""
	values = np.asarray(values, dtype=float)
	if len(np.unique(values)) < 2:
		return float(np.median(values))
	grid = np.linspace(values.min(), values.max(), n_grid)
	return float(grid[np.argmax(gaussian_kde(values)(grid))])


# statistics of a SASA series by name, for choosing a representative frame
STATISTICS: dict[str, Callable[[np.ndarray], float]] = {
	'median': np.median,
	'mean': np.mean,
	'mode': mode,
	'q1': lambda values: np.percentile(values, 25),
	'q3': lambda values: np.percentile(values, 75),
	'min': np.min,
	'max': np.max,
	'mean+std': lambda values: np.mean(values) + np.std(values),
	'mean-std': lambda values: np.mean(values) - np.std(values),
}


def _statistic(statistic: str | Callable[[np.ndarray], float] | float) -> tuple[Callable[[np.ndarray], float], str]:
	"""A statistic (name, function or fixed value) as a function of the values, and its name."""
	if isinstance(statistic, str):
		if statistic not in STATISTICS:
			raise ValueError(f"Unknown statistic {statistic!r}; known: {', '.join(STATISTICS)}, "
			                 f"or give a function or a value.")
		return STATISTICS[statistic], statistic
	if callable(statistic):
		name = getattr(statistic, '__name__', '')
		return statistic, name if name and name != '<lambda>' else 'custom statistic'
	value = float(statistic)
	return (lambda values: value), f"value {value:g}"


def closest_to(values: ArrayLike, statistic: str | Callable[[np.ndarray], float] | float = 'median') -> int:
	"""
	Index of the value closest to a statistic of the values, i.e. a representative frame.

	Parameters
	----------
	values : array-like
		(n,) values, e.g. the SASA of every frame
	statistic : str, callable or float
		name in :data:`STATISTICS` ('median', 'mean', 'mode', 'q1', 'q3', 'min', 'max',
		'mean+std', 'mean-std'),
		a function of the values (e.g. ``lambda v: np.percentile(v, 90)``), or a fixed value
	"""
	values = np.asarray(values)
	function, _ = _statistic(statistic)
	return int(np.argmin(np.abs(values - function(values))))


def best_matching_frame(positions: ArrayLike, reference: ArrayLike) -> tuple[int, np.ndarray, float]:
	"""
	Frame of ``positions`` (n_frames, n_atoms, 3) with the lowest RMSD to
	``reference`` (n_atoms, 3) after optimal superposition, and that frame
	superposed onto the reference.

	Returns
	-------
	frame : int
		index of the best frame in ``positions``
	aligned : np.ndarray
		(n_atoms, 3) positions of the best frame superposed onto ``reference``
	rmsd : float
		RMSD (units of the input) of the aligned frame
	"""
	reference = np.asarray(reference, dtype=np.float64)
	ref_com = reference.mean(axis=0)
	ref = reference - ref_com
	best = (None, np.inf, None)
	for frame, pos in enumerate(positions):
		mobile = np.asarray(pos, dtype=np.float64)
		mobile = mobile - mobile.mean(axis=0)
		rot, rmsd = rotation_matrix(mobile, ref)
		if rmsd < best[1]:
			best = (frame, rmsd, mobile @ np.asarray(rot).T + ref_com)
	frame, rmsd, aligned = best
	return frame, aligned, float(rmsd)


def write_surface_pdb(fname: str | Path, surface: Surface) -> None:
	"""Write a :class:`Surface` as a PDB file, see :meth:`Surface.write`."""
	atoms = surface.atoms
	n_atoms, n_dots = atoms.n_atoms, len(surface.dots)
	resids = atoms.resids if hasattr(atoms, 'resids') else atoms.resindices + 1
	_write_pdb(fname,
	           positions=np.vstack([surface.positions, surface.dots]),
	           names=np.concatenate([atoms.names, ['DOT'] * n_dots]),
	           resnames=np.concatenate([atoms.resnames, atoms.resnames[surface.owners]]),
	           resids=np.concatenate([resids, resids[surface.owners]]),
	           chains=['A'] * n_atoms + ['X'] * n_dots,
	           record_types=['ATOM'] * n_atoms + ['HETATM'] * n_dots,
	           remarks=surface.remarks or [])


def _write_pdb(fname: str | Path, positions: np.ndarray, names, resnames, resids, chains,
               record_types, remarks: list[str]) -> None:
	"""Write per-record arrays as a PDB file with MDAnalysis. Positions are in nm."""
	n_records = len(positions)
	# one residue per consecutive (chain, resid, resname) block
	resindex, res_keys = [], {}
	for key in zip(chains, resids, resnames):
		resindex.append(res_keys.setdefault(key, len(res_keys)))
	universe = mda.Universe.empty(n_records, n_residues=len(res_keys), atom_resindex=resindex,
	                              trajectory=True)
	universe.add_TopologyAttr('names', list(names))
	universe.add_TopologyAttr('resnames', [key[2] for key in res_keys])
	universe.add_TopologyAttr('resids', [int(key[1]) for key in res_keys])
	universe.add_TopologyAttr('chainIDs', list(chains))
	universe.add_TopologyAttr('record_types', list(record_types))
	# explicit defaults for the remaining PDB fields, so the writer does not warn about them
	universe.add_TopologyAttr('altLocs', [''] * n_records)
	universe.add_TopologyAttr('icodes', [''] * len(res_keys))
	universe.add_TopologyAttr('segids', [''])
	universe.add_TopologyAttr('occupancies', np.ones(n_records))
	universe.add_TopologyAttr('tempfactors', np.zeros(n_records))
	universe.add_TopologyAttr('elements', [''] * n_records)
	universe.add_TopologyAttr('formalcharges', np.zeros(n_records, dtype=int))
	universe.atoms.positions = to_angstrom(positions) # MDAnalysis writes Å
	universe.trajectory.remarks = list(remarks)
	with warnings.catch_warnings():
		# a (superposed) molecule has no meaningful box: the unitary CRYST1 record is intended
		warnings.filterwarnings('ignore', message='Unit cell dimensions not found')
		with mda.Writer(str(fname), n_atoms=n_records) as writer:
			writer.write(universe.atoms)
