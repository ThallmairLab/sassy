"""
SASA of every molecule, or of every atom, of an MDAnalysis AtomGroup over a
trajectory.
"""
from collections.abc import Iterator, Mapping
import logging
import time
import numpy as np
from numpy.typing import ArrayLike
import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar

from .dots import unit_dots
from .engine import _exposed_dots
from .groups import molecule_index
from .radii import PROBE_RADIUS, atom_radii

logger = logging.getLogger(__name__)

N_DOTS = 4800 # default minimum number of dots per atom (-> 4842 dots)


class SASA:
	"""
	Solvent accessible surface area of every molecule in an MDAnalysis
	AtomGroup (Shrake-Rupley), optionally per atom.

	"Atom" means every particle of the AtomGroup, as in MDAnalysis: an atom in
	atomistic systems, a coarse-grained bead with ``cg=True``.

	Every molecule (by molnum, or without molnums by detecting consecutive copies,
	see :func:`molecule_index`) is taken on its own, so other molecules do not
	bury its surface. With ``per_molecule=False`` the whole AtomGroup is taken as
	one body instead: all its atoms bury each other. Atoms with radius 0 (dummy beads, virtual sites) are
	ignored. Molecules are assumed to be whole.

	Parameters
	----------
	ag : MDAnalysis.AtomGroup
		atom group for which to calculate the SASA
	cg : bool
		use the CG bead radii instead of the atomistic van der Waals radii
	probe : float
		probe radius (nm)
	ndots : int
		minimum number of dots per atom, see :func:`unit_dots`
	radii : np.ndarray or dict, optional
		van der Waals radii (nm): either an array with one radius per atom in ``ag``, used
		instead of the default radii, or a dict by element (bead size T/S/R/U with ``cg``) that
		adds to or replaces the default radii, e.g. ``{'Na': 0.227}``
	max_frame_atoms : int
		number of atoms (summed over frames) computed together in one parallel call
	per_molecule : bool
		take every molecule on its own (default); False takes the whole AtomGroup as one
		molecule, so that molecules bury each other

	Examples
	--------
	>>> sasa = SASA(universe.select_atoms('moltype GSH'))
	>>> series = sasa.run()  # (n_molecules, n_frames), nm^2
	>>> per_atom = sasa.run(per_atom=True)  # (n_atoms, n_frames), nm^2
	>>> sasa.get_sasa('name S*')  # (n_frames,) SASA of the selection over time
	>>> sasa.get_molecule_sasa(3)  # (n_frames,) SASA of molecule 3 over time

	Attributes
	----------
	sasa : np.ndarray
		(n_atoms, n_frames) SASA (nm^2) of every atom of ``ag`` from the last :meth:`run`
		(None before), atoms in the order of ``ag``
	frames, times : np.ndarray
		(n_frames,) frame numbers and times (ps) of the last :meth:`run`
	"""
	def __init__(self, ag: mda.AtomGroup, cg: bool = False, probe: float = PROBE_RADIUS, ndots: int = N_DOTS,
	             radii: ArrayLike | Mapping[str, float] | None = None, max_frame_atoms: int = 20000,
	             per_molecule: bool = True) -> None:
		
		# prepare radii
		if radii is None or isinstance(radii, Mapping):
			radii = atom_radii(ag, cg=cg, table=radii)
		else:
			radii = np.asarray(radii, dtype=float)
			if radii.shape != (ag.n_atoms,):
				raise ValueError(f"radii must have one value per atom ({ag.n_atoms}), got shape {radii.shape}.")

		# prepare atoms/molecules
		self.ag = ag
		# molecule index of atoms; one molecule for the whole AtomGroup without per_molecule
		molecules = molecule_index(ag) if per_molecule else np.zeros(ag.n_atoms, dtype=int)
		self.n_molecules = int(molecules.max()) + 1 if len(molecules) else 0
		active = np.flatnonzero(radii > 0) # select only atoms with a radius
		self._active = active # positions in ag of the atoms with a radius
		self._indices = ag.indices[active]
		self._molecules = molecules[active]

		# prepare dots
		self.unit = unit_dots(int(ndots))
		self._ext = (radii[active] + probe) * 10.0 # Å
		self._dot_area = 4 * np.pi * (self._ext / 10.0)**2 / len(self.unit) # nm^2 per dot
		
		# every molecule is placed relative to its first atom
		# anchor is the index of the first atom of every molecule in the active atom
		_, first = np.unique(self._molecules, return_index=True)
		anchor = np.zeros(self.n_molecules, dtype=int)
		anchor[self._molecules[first]] = first
		self._anchor = anchor[self._molecules]

		# parallelization
		self._frames_per_block = max(1, max_frame_atoms // max(1, len(active)))

		# results of run(), see get_sasa / get_molecule_sasa
		self._lookup = np.full(ag.universe.atoms.n_atoms, -1, dtype=np.int64) # universe index -> position in ag
		self._lookup[ag.indices] = np.arange(ag.n_atoms)
		self.sasa = None # (n_atoms, n_frames) SASA (nm^2) of every atom of ag
		self.frames = None # (n_frames,) frame numbers
		self.times = None # (n_frames,) times (ps)
		self._molecule_sasa = None
		logger.debug("%d atoms (%d with a radius) in %d %s; %d dots per atom, probe %.3f nm; "
		             "%d frame(s) per parallel call", ag.n_atoms, len(active), self.n_molecules,
		             'molecule(s)' if per_molecule else 'body (whole AtomGroup)', len(self.unit), probe,
		             self._frames_per_block)

	def _atom_areas(self, xyz: np.ndarray) -> np.ndarray:
		"""
		SASA (nm^2) of every atom with a radius in a block of frames.

		Parameters
		----------
		xyz : np.ndarray
			(n_frames, n_atoms, 3) positions (Å) of the atoms with a radius

		Returns
		-------
		np.ndarray
			(n_frames, n_atoms)
		"""
		n_frames, n_atoms, _ = xyz.shape
		
		# put every (frame, molecule) block in its own slab along x and seperate them,
		# so a _neighbor_list finds the neighbours of all of them and no pairs between them
		xyz = xyz - xyz[:, self._anchor]
		span = 2 * np.abs(xyz).max() + 2 * self._ext.max() + 1.0
		block = (np.arange(n_frames)[:, None] * self.n_molecules + self._molecules[None]).ravel()
		xyz = xyz.reshape(-1, 3)
		xyz[:, 0] += block * span
		
		ext = np.tile(self._ext, n_frames)
		exposed = np.count_nonzero(_exposed_dots(xyz, ext, block, self.unit), axis=1)
		areas = exposed.reshape(n_frames, n_atoms) * self._dot_area
		return areas

	def _atom_blocks(self, start: int | None, stop: int | None, step: int | None,
	                 times: list | None = None) -> Iterator[np.ndarray]:
		"""
		Calculate the atom SASA, (n_frames, n_atoms), of blocks of frames of ``start:stop:step``;
		the time of every frame is appended to ``times``.
		"""
		trajectory = self.ag.universe.trajectory
		current = trajectory.ts.frame
		buffer = []
		try:
			for ts in trajectory[start:stop:step]:
				buffer.append(ts.positions[self._indices]) # save positions
				if times is not None:
					times.append(ts.time)
				if len(buffer) == self._frames_per_block: 
					yield self._atom_areas(np.array(buffer, dtype=np.float64)) # run SASA calculation
					buffer = []
			if buffer:
				yield self._atom_areas(np.array(buffer, dtype=np.float64))
		finally:
			trajectory[current] # leave the trajectory on the frame it was on

	
	def _molecule_areas(self, areas: np.ndarray) -> np.ndarray:
		"""(n_frames, n_atoms) atom SASA -> (n_frames, n_molecules) molecule SASA."""
		n_frames = len(areas)
		block = (np.arange(n_frames)[:, None] * self.n_molecules + self._molecules[None]).ravel()
		return np.bincount(block, weights=areas.ravel(),
		                   minlength=n_frames * self.n_molecules).reshape(n_frames, -1)


	def run(self, start: int | None = None, stop: int | None = None, step: int | None = None,
	        per_atom: bool = False, verbose: bool = False, name: str = "") -> np.ndarray:
		"""
		SASA (nm^2) of every molecule, or of every atom, in the frames ``start:stop:step``
		of the trajectory.

		The SASA of every atom is stored in :attr:`sasa` (with :attr:`frames` and
		:attr:`times`), so that afterwards the SASA of any selection or molecule can
		be taken with :meth:`get_sasa` and :meth:`get_molecule_sasa`. A new run
		replaces the stored results.

		Parameters
		----------
		start, stop, step : int, optional
			frames of the trajectory, as in ``trajectory[start:stop:step]``
		per_atom : bool
			return the SASA of every atom instead of every molecule.
			The SASA of an atom is that within its own molecule, so the atoms of a
			molecule sum to its SASA.
		verbose : bool
			show a progress bar over the frames. It advances once per block of frames
			(see ``max_frame_atoms``), so it costs no measurable time but moves in steps.
		name : str
			label of the progress bar

		Returns
		-------
		np.ndarray
			(n_molecules, n_frames), molecules in the order of :func:`split_molecules`, or
			with ``per_atom`` (n_atoms, n_frames), atoms in the order of ``ag``; atoms with
			radius 0 have SASA 0
		"""
		trajectory = self.ag.universe.trajectory
		self.frames = np.arange(trajectory.n_frames)[start:stop:step]
		n_frames = len(self.frames)
		self.sasa = np.zeros((self.ag.n_atoms, n_frames))
		self._molecule_sasa = np.zeros((self.n_molecules, n_frames))
		times = []
		tic = time.perf_counter()
		if len(self._indices) and n_frames: # else only atoms without radius / no frames
			done = 0
			with ProgressBar(total=n_frames, disable=not verbose, desc=f"SASA calculation {name}", unit="frame") as bar:
				for areas in self._atom_blocks(start, stop, step, times):
					frames = slice(done, done + len(areas))
					self.sasa[self._active, frames] = areas.T
					self._molecule_sasa[:, frames] = self._molecule_areas(areas).T
					done += len(areas)
					bar.update(len(areas))
		else:
			current = trajectory.ts.frame
			times = [ts.time for ts in trajectory[start:stop:step]]
			trajectory[current]
		self.times = np.array(times, dtype=float)
		elapsed = time.perf_counter() - tic
		logger.debug("SASA of %d frame(s) in %.2f s (%.1f frames/s)", n_frames, elapsed,
		             n_frames / elapsed if elapsed > 0 else float('inf'))
		return self.sasa if per_atom else self._molecule_sasa

	def _positions(self, selection: str | mda.AtomGroup | None) -> np.ndarray:
		"""Positions in ``ag`` of the atoms of a selection string (applied to ``ag``) or an AtomGroup."""
		if selection is None:
			return np.arange(self.ag.n_atoms)
		if isinstance(selection, str):
			selection = self.ag.select_atoms(selection)
		elif isinstance(selection, mda.core.groups.Atom):
			selection = selection.universe.atoms[[selection.index]]
		elif hasattr(selection, 'atoms'): # AtomGroup, Residue(Group), Segment(Group)
			selection = selection.atoms
		else:
			raise TypeError(f"selection must be a selection string or an AtomGroup, got {type(selection).__name__}.")
		if selection.universe is not self.ag.universe:
			raise ValueError("The selection belongs to a different Universe than the AtomGroup of SASA.")
		positions = self._lookup[selection.indices]
		if (positions < 0).any():
			missing = selection.indices[positions < 0]
			raise ValueError(f"{len(missing)} atom(s) of the selection are not in the AtomGroup of SASA: "
			                 f"universe indices {missing[:10].tolist()}{' ...' if len(missing) > 10 else ''}.")
		return positions

	def _check_run(self) -> None:
		if self.sasa is None:
			raise RuntimeError("No SASA computed yet: call run() first.")

	def get_sasa(self, selection: str | mda.AtomGroup | None = None, per_atom: bool = False) -> np.ndarray:
		"""
		SASA (nm^2) over time of a selection of the atoms of ``ag``, from the last :meth:`run`.

		The SASA of every atom is that within its own (whole) molecule, so the SASA
		of a selection is the part of the molecules' surface that belongs to its
		atoms, not the SASA of the selected atoms on their own.

		Parameters
		----------
		selection : str or MDAnalysis.AtomGroup, optional
			selection string, applied to ``ag`` (e.g. ``'name C*'``, ``'resid 1'``), or an
			AtomGroup whose atoms are all in ``ag``; default all atoms of ``ag``
		per_atom : bool
			return the SASA of every selected atom instead of their sum

		Returns
		-------
		np.ndarray
			(n_frames,) summed SASA, or with ``per_atom`` (n_selected, n_frames), atoms
			in the order of the selection

		Examples
		--------
		>>> sasa.run()
		>>> sasa.get_sasa('name S*')         # SASA of the sulfur atoms over time
		>>> sasa.get_sasa(u.select_atoms('resname GLY'), per_atom=True)
		"""
		self._check_run()
		atoms = self.sasa[self._positions(selection)]
		return atoms if per_atom else atoms.sum(axis=0)

	def get_molecule_sasa(self, index: int | ArrayLike | None = None) -> np.ndarray:
		"""
		SASA (nm^2) over time of molecules, by molecule index (0, 1, ... in the order
		of :func:`split_molecules`), from the last :meth:`run`.

		Parameters
		----------
		index : int or array-like of int, optional
			molecule index or indices (negative indices count from the end);
			default all molecules

		Returns
		-------
		np.ndarray
			(n_frames,) for a single index, else (n_indices, n_frames)

		Examples
		--------
		>>> sasa.get_molecule_sasa(0)         # first molecule
		>>> sasa.get_molecule_sasa([0, 2])    # first and third molecule
		"""
		self._check_run()
		if index is None:
			return self._molecule_sasa
		index = np.asarray(index)
		if not np.issubdtype(index.dtype, np.integer):
			raise TypeError(f"Molecule indices must be integers, got {index.dtype}.")
		if ((index < -self.n_molecules) | (index >= self.n_molecules)).any():
			raise IndexError(f"Molecule index out of range for {self.n_molecules} molecule(s): {index.tolist()}.")
		return self._molecule_sasa[index]

