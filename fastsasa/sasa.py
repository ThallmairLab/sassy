"""
SASA of every molecule, or of every atom, of an MDAnalysis AtomGroup over a
trajectory.
"""
from collections.abc import Callable, Iterator, Mapping
import logging
import time
import numpy as np
from numpy.typing import ArrayLike
import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar

from .dots import unit_dots
from .engine import _exposed_dots
from .groups import molecule_index, molecule_labels, molecule_types
from .radii import PROBE_RADIUS, atom_radii
from .units import to_nm
from .surface import SURFACE_DOTS, Surface, _statistic, generate_surface

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
		overlap mode: True ("molecule", default) takes every molecule on its own, so only
		atoms of the same molecule bury each other; False ("whole") takes the whole
		AtomGroup as one body, so all its atoms bury each other
	save_atoms : bool
		keep the SASA of every atom of every frame (:attr:`sasa`, (n_atoms, n_frames)),
		needed for :meth:`get_sasa` of selections and ``run(per_atom=True)``. The SASA of
		every molecule and the total are always kept; False saves the memory of the
		per-atom array for very large systems or long trajectories

	Examples
	--------
	>>> sasa = SASA(universe.select_atoms('moltype GSH'))
	>>> series = sasa.run()  # (n_molecules, n_frames), nm^2
	>>> per_atom = sasa.run(per_atom=True)  # (n_atoms, n_frames), nm^2
	>>> sasa.get_sasa('name S*')  # (n_frames,) SASA of the selection over time
	>>> sasa.get_molecule_sasa(3)  # (n_frames,) SASA of molecule 3 over time
	>>> sasa.write_surface('dots.pdb', molecule=0)  # structure + surface dots of a median-SASA frame
	>>> sasa.write_surface('dots.pdb', statistic='mean+std')  # frame closest to mean + std of the SASA

	Attributes
	----------
	sasa : np.ndarray
		(n_atoms, n_frames) SASA (nm^2) of every atom of ``ag`` from the last :meth:`run`,
		atoms in the order of ``ag``; None before a run and with ``save_atoms=False``
	molecule_types : list of str
		(n_molecules,) type of every molecule: its moltype (tpr), else named by its first
		residue (see :func:`molecule_types`); 'selection' without ``per_molecule``
	molecule_labels : list of str
		(n_molecules,) label of every molecule: its type numbered within the type (GSH1,
		GSH2, SOL1, ...); 'selection' without ``per_molecule``
	types : dict of str to np.ndarray
		molecule indices of every type, in the order of first appearance
	molecule_sasa : np.ndarray
		(n_molecules, n_frames) SASA (nm^2) of every molecule from the last :meth:`run`
	total_sasa : np.ndarray
		(n_frames,) total SASA (nm^2) of ``ag`` from the last :meth:`run`
	frames, times : np.ndarray
		(n_frames,) frame numbers and times (ps) of the last :meth:`run`
	"""
	def __init__(self, ag: mda.AtomGroup, cg: bool = False, probe: float = PROBE_RADIUS, ndots: int = N_DOTS,
	             radii: ArrayLike | Mapping[str, float] | None = None, max_frame_atoms: int = 20000,
	             per_molecule: bool = True, save_atoms: bool = True) -> None:

		# prepare radii
		self._custom_radii = radii is not None # for the remarks of written surfaces
		if radii is None or isinstance(radii, Mapping):
			radii = atom_radii(ag, cg=cg, table=radii)
		else:
			radii = np.asarray(radii, dtype=float)
			if radii.shape != (ag.n_atoms,):
				raise ValueError(f"radii must have one value per atom ({ag.n_atoms}), got shape {radii.shape}.")

		self.cg = cg
		self.probe = probe
		self.per_molecule = per_molecule
		self.save_atoms = save_atoms

		# prepare atoms/molecules
		self.ag = ag
		self.radii = radii # (n_atoms,) van der Waals radii (nm)
		# molecule index of atoms; one molecule for the whole AtomGroup without per_molecule
		molecules = molecule_index(ag) if per_molecule else np.zeros(ag.n_atoms, dtype=int)
		self.n_molecules = int(molecules.max()) + 1 if len(molecules) else 0
		self._molecule_of_atom = molecules
		# type and label of every molecule, and the molecules of every type (in order of appearance)
		self.molecule_types = molecule_types(ag, molecules) if per_molecule else ['selection'] * self.n_molecules
		self.molecule_labels = molecule_labels(self.molecule_types) if per_molecule else self.molecule_types[:]
		self.types = {}
		for i, name in enumerate(self.molecule_types):
			self.types.setdefault(name, []).append(i)
		self.types = {name: np.array(indices) for name, indices in self.types.items()}
		active = np.flatnonzero(radii > 0) # select only atoms with a radius
		self._active = active # indices in ag of the atoms with a radius
		self._indices = ag.indices[active]
		self._molecules = molecules[active]

		# prepare dots
		self.unit = unit_dots(int(ndots))
		self._ext = radii[active] + probe # expanded radii (nm)
		self._dot_area = 4 * np.pi * self._ext**2 / len(self.unit) # nm^2 per dot
		
		# every molecule is placed relative to its first atom
		# anchor is the index of the first atom of every molecule in the active atom
		_, first = np.unique(self._molecules, return_index=True)
		anchor = np.zeros(self.n_molecules, dtype=int)
		anchor[self._molecules[first]] = first
		self._anchor = anchor[self._molecules]

		# parallelization
		self._frames_per_block = max(1, max_frame_atoms // max(1, len(active)))

		# prep results of run(), see get_sasa / get_molecule_sasa
		self._lookup = np.full(ag.universe.atoms.n_atoms, -1, dtype=np.int64) # universe index -> index in ag
		self._lookup[ag.indices] = np.arange(ag.n_atoms)
		self.sasa = None # (n_atoms, n_frames) SASA (nm^2) of every atom of ag, only with save_atoms
		self.molecule_sasa = None # (n_molecules, n_frames) SASA (nm^2) of every molecule
		self.total_sasa = None # (n_frames,) total SASA (nm^2) of ag
		self.frames = None # (n_frames,) frame numbers
		self.times = None # (n_frames,) times (ps)
		logger.debug("%d atoms (%d with a radius) in %d %s; %d dots per atom, probe %.3f nm; "
		             "%d frame(s) per parallel call", ag.n_atoms, len(active), self.n_molecules,
		             'molecule(s)' if per_molecule else 'body (whole AtomGroup)', len(self.unit), probe,
		             self._frames_per_block)

	# SASA calculation
	def _atom_areas(self, xyz: np.ndarray) -> np.ndarray:
		"""
		SASA (nm^2) of every atom in a block of frames.

		Parameters
		----------
		xyz : np.ndarray
			(n_frames, n_atoms, 3) positions (nm) of the atoms with a radius

		Returns
		-------
		np.ndarray
			(n_frames, n_atoms)
		"""
		n_frames, n_atoms, _ = xyz.shape
		
		# put every (frame, molecule) block in its own slab along x and seperate them,
		# so a _neighbor_list finds the neighbours of all of them and no pairs between them
		xyz = xyz - xyz[:, self._anchor]
		span = 2 * np.abs(xyz).max() + 2 * self._ext.max() + 0.1 # nm
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
				buffer.append(to_nm(ts.positions[self._indices])) # save positions (nm)
				if times is not None:
					times.append(ts.time)
				if len(buffer) == self._frames_per_block: 
					yield self._atom_areas(np.array(buffer)) # run SASA calculation
					buffer = []
			if buffer:
				yield self._atom_areas(np.array(buffer))
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

		The SASA of every molecule and the total are stored in :attr:`molecule_sasa`
		and :attr:`total_sasa`, and (with ``save_atoms``) that of every atom in
		:attr:`sasa`, with :attr:`frames` and :attr:`times`, so that afterwards the SASA
		of any selection or molecule can be taken with :meth:`get_sasa` and
		:meth:`get_molecule_sasa`. A new run replaces the stored results.

		Parameters
		----------
		start, stop, step : int, optional
			frames of the trajectory, as in ``trajectory[start:stop:step]``
		per_atom : bool
			return the SASA of every atom instead of every molecule (needs ``save_atoms``).
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
		if per_atom and not self.save_atoms:
			raise ValueError("per_atom needs the SASA of every atom: create the SASA with save_atoms=True.")
		trajectory = self.ag.universe.trajectory
		frames = np.arange(trajectory.n_frames)[start:stop:step]
		n_frames = len(frames)
		atom_sasa = np.zeros((self.ag.n_atoms, n_frames)) if self.save_atoms else None
		molecule_sasa = np.zeros((self.n_molecules, n_frames))
		times = []
		tic = time.perf_counter()
		if len(self._indices) and n_frames: # else only atoms without radius / no frames
			done = 0
			with ProgressBar(total=n_frames, disable=not verbose, desc=f"SASA calculation {name}", unit="frame") as bar:
				for areas in self._atom_blocks(start, stop, step, times):
					block = slice(done, done + len(areas))
					if atom_sasa is not None:
						atom_sasa[self._active, block] = areas.T
					molecule_sasa[:, block] = self._molecule_areas(areas).T
					done += len(areas)
					bar.update(len(areas))
		else:
			current = trajectory.ts.frame
			times = [ts.time for ts in trajectory[start:stop:step]]
			trajectory[current]
		# results only replaced after a complete run
		self.frames, self.times = frames, np.array(times, dtype=float)
		self.sasa, self.molecule_sasa = atom_sasa, molecule_sasa
		self.total_sasa = molecule_sasa.sum(axis=0)
		elapsed = time.perf_counter() - tic
		logger.debug("SASA of %d frame(s) in %.2f s (%.1f frames/s)", n_frames, elapsed,
		             n_frames / elapsed if elapsed > 0 else float('inf'))
		return self.sasa if per_atom else self.molecule_sasa

	# SASA retrieval
	def _ag_indices(self, selection: str | mda.AtomGroup | None) -> np.ndarray:
		"""Indices in ``ag`` of the atoms of a selection string (applied to ``ag``) or an AtomGroup."""
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
		indices = self._lookup[selection.indices]
		if (indices < 0).any():
			missing = selection.indices[indices < 0]
			raise ValueError(f"{len(missing)} atom(s) of the selection are not in the AtomGroup of SASA: "
			                 f"universe indices {missing[:10].tolist()}{' ...' if len(missing) > 10 else ''}.")
		return indices

	def _check_run(self, atoms: bool = False) -> None:
		'''
		Check that a SASA has been computed with SASA.run, and that the SASA of every atom is available if ``atoms``.
		  Raises a RuntimeError otherwise.
		'''
		if self.frames is None:
			raise RuntimeError("No SASA computed yet: call run() first.")
		if atoms and self.sasa is None:
			raise RuntimeError("The SASA of every atom was not saved (save_atoms=False): only the total "
			                   "and molecule SASA are available.")

	def get_sasa(self, selection: str | mda.AtomGroup | None = None, per_atom: bool = False) -> np.ndarray:
		"""
		SASA (nm^2) over time of a selection of the atoms of ag, from the last SASA.run().

		The SASA of every atom is that within its own (whole) molecule, so the SASA
		of a selection is the part of the molecules' surface that belongs to its
		atoms, not the SASA of the selected atoms on their own.

		Parameters
		----------
		selection : str or MDAnalysis.AtomGroup, optional
			selection string, applied to ``ag`` (e.g. ``'name C*'``, ``'resid 1'``), or an
			AtomGroup whose atoms are all in ``ag``; default all atoms of ``ag`` (the total,
			also without ``save_atoms``; selections need ``save_atoms``)
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
		if selection is None and not per_atom:
			self._check_run()
			return self.total_sasa
		self._check_run(atoms=True)
		atoms = self.sasa[self._ag_indices(selection)]
		return atoms if per_atom else atoms.sum(axis=0)

	def molecule_number(self, molecule: int | str) -> int:
		"""
		Index (0, 1, ...) of a molecule given by its index (negative counts from the end) or
		its label (e.g. 'GSH2', see :attr:`molecule_labels`).
		"""
		if isinstance(molecule, str):
			if molecule not in self.molecule_labels:
				raise KeyError(f"No molecule {molecule!r}; labels: {', '.join(self.molecule_labels[:10])}"
				               f"{', ...' if self.n_molecules > 10 else ''}.")
			return self.molecule_labels.index(molecule)
		if not -self.n_molecules <= molecule < self.n_molecules:
			raise IndexError(f"Molecule index {molecule} out of range for {self.n_molecules} molecule(s).")
		return int(molecule) % self.n_molecules

	def get_type_sasa(self, name: str, total: bool = False) -> np.ndarray:
		"""
		SASA (nm^2) over time of the molecules of one type (see :attr:`types`), from the
		last SASA.run(): (n_molecules_of_type, n_frames), or with ``total`` their sum (n_frames,).
		"""
		self._check_run()
		if name not in self.types:
			raise KeyError(f"No molecule type {name!r}; types: {', '.join(self.types)}.")
		values = self.molecule_sasa[self.types[name]]
		return values.sum(axis=0) if total else values

	def get_molecule_sasa(self, index: int | str | ArrayLike | None = None) -> np.ndarray:
		"""
		SASA (nm^2) over time of molecules, by molecule index (0, 1, ... in the order
		of :func:`split_molecules`) or label (see :attr:`molecule_labels`), from the last
		SASA.run().

		Parameters
		----------
		index : int, str or array-like of int, optional
			molecule index or indices (negative indices count from the end), or a label
			(e.g. 'GSH2'); default all molecules

		Returns
		-------
		np.ndarray
			(n_frames,) for a single index, else (n_indices, n_frames)

		Examples
		--------
		>>> sasa.get_molecule_sasa(0)         # first molecule
		>>> sasa.get_molecule_sasa([0, 2])    # first and third molecule
		>>> sasa.get_molecule_sasa('GSH2')    # by label
		"""
		self._check_run()
		if index is None:
			return self.molecule_sasa
		if isinstance(index, str):
			return self.molecule_sasa[self.molecule_number(index)]
		index = np.asarray(index)
		if not np.issubdtype(index.dtype, np.integer):
			raise TypeError(f"Molecule indices must be integers, got {index.dtype}.")
		if ((index < -self.n_molecules) | (index >= self.n_molecules)).any():
			raise IndexError(f"Molecule index out of range for {self.n_molecules} molecule(s): {index.tolist()}.")
		return self.molecule_sasa[index]


	# SASA surface visualization
	def molecule_atoms(self, molecule: int | str | None = None) -> mda.AtomGroup:
		"""
		Atoms of a molecule (index as in :meth:`get_molecule_sasa`, negative counts from
		the end, or label); all atoms of ``ag`` for None.
		"""
		if molecule is None:
			return self.ag
		return self.ag[self._molecule_of_atom == self.molecule_number(molecule)]

	def _molecule_remark(self, atoms: mda.AtomGroup, molecule: int) -> str:
		"""REMARK record naming the atoms of a written surface: which molecule, or the whole AtomGroup."""
		name = atoms.moltypes[0] if hasattr(atoms, 'moltypes') else (
		       '/'.join(dict.fromkeys(atoms.resnames)) if hasattr(atoms, 'resnames') else 'molecule')
		resids = atoms.resids if hasattr(atoms, 'resids') else atoms.resindices + 1
		residues = f"resid {resids.min()}" if resids.min() == resids.max() else f"resids {resids.min()}-{resids.max()}"
		described = f"{name}, {residues}, {atoms.n_atoms} atoms"
		if not self.per_molecule:
			return f"Whole selection as one body ({described})"
		return f"Molecule {molecule % self.n_molecules} of {self.n_molecules} ({described}), on its own"

	def representative(self, molecule_type: str, statistic: str | Callable[[np.ndarray], float] | float = 'median',
	                   frame: int | None = None) -> tuple[int, int, float]:
		"""
		A representative molecule of a type and frame: the statistic is taken over the SASA
		distribution of that type only (all its molecules in all analysed frames of the last
		:meth:`run`), and the (molecule, frame) whose SASA is closest to it is chosen.

		Parameters
		----------
		molecule_type : str
			type (see :attr:`types`)
		statistic : str, callable or float
			see :meth:`write_surface`
		frame : int, optional
			fixed trajectory frame: only the molecule is chosen (the first of the type if the
			frame was not analysed)

		Returns
		-------
		molecule : int
			molecule index
		frame : int
			trajectory frame
		target : float
			the statistic (nm^2) of the type's distribution
		"""
		self._check_run()
		if molecule_type not in self.types:
			raise KeyError(f"No molecule type {molecule_type!r}; types: {', '.join(self.types)}.")
		function, _ = _statistic(statistic)
		molecules = self.types[molecule_type]
		values = self.molecule_sasa[molecules] # (n_molecules_of_type, n_frames)
		target = float(function(values.ravel()))
		if frame is None:
			k, column = np.unravel_index(np.argmin(np.abs(values - target)), values.shape)
			return int(molecules[k]), int(self.frames[column]), target
		frame = int(frame) % self.ag.universe.trajectory.n_frames
		analysed = np.flatnonzero(self.frames == frame)
		if not len(analysed):
			return int(molecules[0]), frame, target
		return int(molecules[np.argmin(np.abs(values[:, analysed[0]] - target))]), frame, target

	def write_surface(self, fname: str, frame: int | None = None, molecule: int | str = 0,
	                  reference: ArrayLike | mda.AtomGroup | None = None,
	                  ndots: int = SURFACE_DOTS, return_surface: bool = False,
	                  statistic: str | Callable[[np.ndarray], float] | float = 'median',
	                  molecule_type: str | None = None) -> Surface | None:
		"""
		Write the structure and its solvent accessible surface dots (Connolly-style dot
		surface) in one frame to a PDB file, with the radii and probe of this SASA; see
		:func:`generate_surface` and :meth:`Surface.write` (format).

		The surface follows the overlap mode of the SASA: with ``per_molecule`` it is the
		surface of a single molecule (on its own), else that of the whole AtomGroup.

		Statistics are taken over the SASA distribution of the molecule's type only (all
		molecules of the type in all analysed frames; whole mode: the total), so every type
		has its own median, mean, ...

		Parameters
		----------
		fname : str
			PDB file to write
		frame : int, optional
			trajectory frame (negative counts from the end); default the analysed frame of the
			molecule whose SASA is closest to ``statistic`` (by default the median) of its
			type's distribution, i.e. a representative structure
		molecule : int or str
			with ``per_molecule``: index of the molecule (as in :meth:`get_molecule_sasa`,
			negative counts from the end) or its label (e.g. 'GSH2'), default the first;
			without: the whole AtomGroup is the only molecule (0)
		molecule_type : str, optional
			instead of ``molecule``: a type (see :attr:`types`); its representative molecule
			and frame are used, see :meth:`representative` (with ``frame``: the molecule
			closest to the statistic in that frame; with ``reference``: its first molecule)
		reference : np.ndarray or MDAnalysis.AtomGroup, optional
			reference structure, (n_atoms, 3) positions (nm) or an AtomGroup with as many
			atoms as the surface (in the same order, e.g. a mapped atomistic structure for
			a CG model): instead of ``frame``, the frame (of the last :meth:`run`, else of
			the whole trajectory) with the lowest RMSD to it is used, superposed onto it, so
			that both surfaces share one coordinate frame
		ndots : int
			minimum number of dots per atom (fewer than for the SASA, as dots are for viewing)
		statistic : str, callable or float
			without ``frame`` and ``reference``: the representative frame is the one whose SASA
			is closest to this statistic of the type's SASA distribution: a name ('median',
			'mean', 'mode' (most probable SASA), 'q1', 'q3' (quartiles), 'min', 'max',
			'mean+std', 'mean-std', see :data:`STATISTICS`), a function of the SASA values
			(e.g. ``lambda sasa: np.percentile(sasa, 90)``), or a fixed SASA (nm^2)
		return_surface : bool
			return the written :class:`Surface` (structure, dots, area in nm^2 from the
			dots, frame, ...) instead of None

		Returns
		-------
		Surface or None
			the written surface with ``return_surface``, else None
		"""
		function, name = _statistic(statistic) # checked also when not used
		chosen = None # remark on how the frame was chosen
		if molecule_type is not None:
			if molecule_type not in self.types:
				raise KeyError(f"No molecule type {molecule_type!r}; types: {', '.join(self.types)}.")
			if reference is not None:
				molecule = int(self.types[molecule_type][0])
			else:
				molecule, frame, target = self.representative(molecule_type, function, frame)
				chosen = self._chosen_remark(molecule, frame, target, name, molecule_choice=True)
		else:
			if not self.per_molecule and molecule not in (0, -1, 'selection'):
				raise ValueError(f"Molecule {molecule}: without per_molecule the surface is that of the whole "
				                 f"AtomGroup (molecule 0).")
			molecule = self.molecule_number(molecule)
			if frame is None and reference is None:
				self._check_run()
				target = float(function(self.get_type_sasa(self.molecule_types[molecule]).ravel()))
				series = self.get_molecule_sasa(molecule)
				frame = int(self.frames[np.argmin(np.abs(series - target))])
				chosen = self._chosen_remark(molecule, frame, target, name)
		atoms = self.molecule_atoms(molecule) # whole mode: all of ag
		indices = self._lookup[atoms.indices]
		result = generate_surface(atoms, radii=self.radii[indices] if self._custom_radii else None,
		                          probe=self.probe, block=self._molecule_of_atom[indices], frame=frame,
		                          reference=reference, frames=self.frames, ndots=ndots, cg=self.cg)
		result.remarks.insert(2, self._molecule_remark(atoms, molecule)) # after the frame
		if chosen is not None:
			result.remarks.insert(3, chosen)
		result.molecule = molecule
		result.write(fname)
		logger.info("wrote the surface of %s in frame %d (%d atoms, %d dots, SASA %.3f nm^2) to %s",
		            self.molecule_labels[molecule], result.frame, result.atoms.n_atoms, len(result.dots),
		            result.area, fname)
		return result if return_surface else None

	def _chosen_remark(self, molecule: int, frame: int, target: float, name: str,
	                   molecule_choice: bool = False) -> str:
		"""REMARK on how the frame (and molecule) of a surface was chosen; also logged."""
		molecule_type = self.molecule_types[molecule]
		n_molecules = len(self.types[molecule_type])
		analysed = np.flatnonzero(self.frames == frame)
		sasa = f"{self.molecule_sasa[molecule, analysed[0]]:.3f} nm^2" if len(analysed) else "not analysed"
		pool = (f"{n_molecules} {molecule_type} molecule(s) x {len(self.frames)} analysed frame(s)"
		        if self.per_molecule else f"{len(self.frames)} analysed frame(s)")
		what = (f"Representative molecule and frame: {self.molecule_labels[molecule]} in frame {frame}"
		        if molecule_choice else "Representative frame")
		logger.info("%s %s in frame %d: SASA %s, closest to the %s (%.3f nm^2) of %s",
		            'representative' if molecule_choice else 'surface of', self.molecule_labels[molecule], frame,
		            sasa, name, target, pool)
		return f"{what}: SASA ({sasa}) closest to the {name} ({target:.3f} nm^2) of {pool}"
