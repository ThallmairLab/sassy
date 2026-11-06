"""
SASA of every molecule, or of every atom, of an MDAnalysis AtomGroup over a
trajectory.
"""
from collections.abc import Iterator, Mapping
import numpy as np
from numpy.typing import ArrayLike
import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar

from .dots import unit_dots
from .engine import _exposed_dots
from .groups import molecule_index
from .radii import PROBE_RADIUS, atom_radii

N_DOTS = 4800 # default minimum number of dots per atom (-> 4842 dots)


class SASA:
	"""
	Solvent accessible surface area of every molecule in an MDAnalysis
	AtomGroup (Shrake-Rupley), optionally per atom.

	"Atom" means every particle of the AtomGroup, as in MDAnalysis: an atom in
	atomistic systems, a coarse-grained bead with ``cg=True``.

	Every molecule (by molnum, or without molnums by detecting consecutive copies,
	see :func:`molecule_index`) is taken on its own, so other molecules do not
	bury its surface. Atoms with radius 0 (dummy beads, virtual sites) are
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

	Examples
	--------
	>>> sasa = SASA(universe.select_atoms('moltype GSH'))
	>>> series = sasa.run()  # (n_molecules, n_frames), nm^2
	>>> per_atom = sasa.run(per_atom=True)  # (n_atoms, n_frames), nm^2
	"""
	def __init__(self, ag: mda.AtomGroup, cg: bool = False, probe: float = PROBE_RADIUS, ndots: int = N_DOTS,
	             radii: ArrayLike | Mapping[str, float] | None = None, max_frame_atoms: int = 20000,
	             ) -> None:
		
		# prepare radii
		if radii is None or isinstance(radii, Mapping):
			radii = atom_radii(ag, cg=cg, table=radii)
		else:
			radii = np.asarray(radii, dtype=float)
			if radii.shape != (ag.n_atoms,):
				raise ValueError(f"radii must have one value per atom ({ag.n_atoms}), got shape {radii.shape}.")

		# prepare atoms/molecules
		self.ag = ag
		molecules = molecule_index(ag) # molecule index of atoms
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

	def _atom_blocks(self, start: int | None, stop: int | None, step: int | None) -> Iterator[np.ndarray]:
		"""Claculate the atom SASA, (n_frames, n_atoms), of blocks of frames of ``start:stop:step``."""
		trajectory = self.ag.universe.trajectory
		current = trajectory.ts.frame
		buffer = []
		try:
			for ts in trajectory[start:stop:step]:
				buffer.append(ts.positions[self._indices]) # save positions
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
	        per_atom: bool = False, verbose: bool = False) -> np.ndarray:
		"""
		SASA (nm^2) of every molecule, or of every atom, in the frames ``start:stop:step``
		of the trajectory.

		Parameters
		----------
		start, stop, step : int, optional
			frames of the trajectory, as in ``trajectory[start:stop:step]``
		per_atom : bool
			return the SASA of every atom instead of every molecule; it is computed
			anyway, so this costs no extra time, only memory for the larger result.
			The SASA of an atom is that within its own molecule, so the atoms of a
			molecule sum to its SASA.
		verbose : bool
			show a progress bar over the frames. It advances once per block of frames
			(see ``max_frame_atoms``), so it costs no measurable time but moves in steps.

		Returns
		-------
		np.ndarray
			(n_molecules, n_frames), molecules in the order of :func:`split_molecules`, or
			with ``per_atom`` (n_atoms, n_frames), atoms in the order of ``ag``; atoms with
			radius 0 have SASA 0
		"""
		n_frames = len(self.ag.universe.trajectory[start:stop:step])
		result = np.zeros((self.ag.n_atoms if per_atom else self.n_molecules, n_frames))
		if not len(self._indices) or not n_frames: # only atoms without radius / no frames
			return result
		done = 0
		with ProgressBar(total=n_frames, disable=not verbose, desc="SASA", unit="frame") as bar:
			for areas in self._atom_blocks(start, stop, step):
				frames = slice(done, done + len(areas))
				if per_atom:
					result[self._active, frames] = areas.T
				else:
					result[:, frames] = self._molecule_areas(areas).T
				done += len(areas)
				bar.update(len(areas))
		return result

