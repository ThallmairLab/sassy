"""
SASA of every molecule (residue, or the whole selection), or of every atom, of an
MDAnalysis AtomGroup over a trajectory, optionally covered by an environment.
"""
from collections.abc import Callable, Iterator, Mapping
import logging
import time
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike
import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar

from .dots import unit_dots
from .engine import _exposed_counts, warn_coincident
from .environment import box_nm, environment_copies
from .groups import (atom_label, molecule_index, molecule_labels, molecule_members, molecule_types,
                     residue_index, residue_labels, residue_types)
from .radii import PROBE_RADIUS, atom_radii
from .units import to_nm
from .surface import SURFACE_DOTS, Surface, _statistic, generate_surface

logger = logging.getLogger(__name__)

N_DOTS = 4800 # default minimum number of dots per atom (-> 4842 dots)

BODIES = ('molecule', 'residue', 'selection') # how the AtomGroup is split into bodies, see SASA
Bodies = Literal['molecule', 'residue', 'selection']


def partition(ag: mda.AtomGroup, bodies: Bodies = 'molecule') -> tuple[np.ndarray, list[str], list[str]]:
	"""
	Split an AtomGroup into bodies: sets of atoms that bury each other.

	Parameters
	----------
	ag : MDAnalysis.AtomGroup
	bodies : str
		'molecule': every molecule (see :func:`molecule_index`); 'residue': every residue
		(see :func:`residue_index`); 'selection': the whole AtomGroup as one body

	Returns
	-------
	index : np.ndarray
		(n_atoms,) body of every atom, 0, 1, ... in order of appearance
	types : list of str
		(n_bodies,) type of every body (molecule type, residue type, or 'selection')
	labels : list of str
		(n_bodies,) unique label of every body (e.g. GSH2, ALA12, 'selection')
	"""
	if bodies == 'molecule':
		index = molecule_index(ag)
		types = molecule_types(ag, index)
		return index, types, molecule_labels(types)
	if bodies == 'residue':
		index = residue_index(ag)
		return index, residue_types(ag, index), residue_labels(ag, index)
	if bodies == 'selection':
		return np.zeros(ag.n_atoms, dtype=int), ['selection'] * bool(ag.n_atoms), ['selection'] * bool(ag.n_atoms)
	raise ValueError(f"bodies must be one of {', '.join(BODIES)}, got {bodies!r}.")


class SASA:
	"""
	Solvent accessible surface area of every molecule in an MDAnalysis
	AtomGroup (Shrake-Rupley), optionally per atom.

	"Atom" means every particle of the AtomGroup, as in MDAnalysis: an atom in
	atomistic systems, a coarse-grained bead with ``cg=True``.

	Two independent settings decide which atoms bury the surface of an atom:

	- ``bodies``: the AtomGroup is split into bodies, and only atoms of the same body
	  bury each other. By default every molecule (by molnum, else by bonds or
	  consecutive copies, see :func:`molecule_index`) is a body, so other molecules do
	  not bury its surface; 'residue' makes every residue a body, 'selection' the whole
	  AtomGroup (all its atoms bury each other).
	- ``environment``: atoms outside the AtomGroup (e.g. a protein around a ligand)
	  that bury every body, but whose own SASA is not computed. Every body is covered
	  by the environment, not by the other bodies (unless they are one body).

	Throughout, "molecule" in names (:attr:`molecule_sasa`, :meth:`get_molecule_sasa`,
	...) means a body: a molecule, a residue with ``bodies='residue'``, or the whole
	AtomGroup (label 'selection') with ``bodies='selection'``.

	Atoms with radius 0 (dummy beads, virtual sites) are ignored. Bodies are assumed
	to be whole; the environment is taken in the periodic image of every body.

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
	bodies : str
		which atoms bury each other: 'molecule' (default, every molecule on its own),
		'residue' (every residue on its own) or 'selection' (the whole AtomGroup as one body)
	environment : str or MDAnalysis.AtomGroup, optional
		atoms that bury every body without being counted (e.g. the protein around a
		ligand): an AtomGroup of the same Universe, or a selection string applied to it;
		must not overlap ``ag``. Its radii are the default ones (with a ``radii`` dict
		applied; a ``radii`` array covers ``ag`` only); atoms with radius 0 are ignored
	buried : bool
		also compute the SASA without the environment, so that the area buried by the
		environment (without minus with it) is available, see :meth:`get_buried_sasa`;
		needs an ``environment``
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
	>>> ligand = SASA(universe.select_atoms('resname LIG'), environment='protein', buried=True)
	>>> ligand.run()  # SASA of the ligand left uncovered by the protein
	>>> ligand.get_buried_sasa()  # (n_frames,) area of the ligand buried by the protein

	Attributes
	----------
	sasa : np.ndarray
		(n_atoms, n_frames) SASA (nm^2) of every atom of ``ag`` from the last :meth:`run`,
		atoms in the order of ``ag``; None before a run and with ``save_atoms=False``
	bodies : str
		how the AtomGroup is split into bodies ('molecule', 'residue' or 'selection')
	environment : MDAnalysis.AtomGroup or None
		the environment atoms with a radius, None without an environment
	molecule_types : list of str
		(n_molecules,) type of every body: its moltype (tpr), else named by its first
		residue (see :func:`molecule_types`); the residue type (see :func:`residue_types`)
		with ``bodies='residue'``; 'selection' with ``bodies='selection'``
	molecule_labels : list of str
		(n_molecules,) label of every body: its type numbered within the type (GSH1,
		GSH2, SOL1, ...); ``<resname><resid>`` (ALA12) with ``bodies='residue'``;
		'selection' with ``bodies='selection'``
	types : dict of str to np.ndarray
		molecule indices of every type, in the order of first appearance
	molecule_sasa : np.ndarray
		(n_molecules, n_frames) SASA (nm^2) of every molecule from the last :meth:`run`
	total_sasa : np.ndarray
		(n_frames,) total SASA (nm^2) of ``ag`` from the last :meth:`run`
	frames, times : np.ndarray
		(n_frames,) frame numbers and times (ps) of the last :meth:`run`
	free_sasa, free_molecule_sasa, free_total_sasa : np.ndarray
		with ``buried``: as :attr:`sasa`, :attr:`molecule_sasa` and :attr:`total_sasa`,
		but without the environment; None otherwise
	"""
	def __init__(self, ag: mda.AtomGroup, cg: bool = False, probe: float = PROBE_RADIUS, ndots: int = N_DOTS,
	             radii: ArrayLike | Mapping[str, float] | None = None, max_frame_atoms: int = 20000,
	             bodies: Bodies = 'molecule', environment: str | mda.AtomGroup | None = None,
	             buried: bool = False, save_atoms: bool = True) -> None:

		if bodies not in BODIES:
			raise ValueError(f"bodies must be one of {', '.join(BODIES)}, got {bodies!r}.")
		if not probe >= 0:
			raise ValueError(f"The probe radius must be >= 0 (nm), got {probe}.")
		if buried and environment is None:
			raise ValueError("buried needs an environment: the area buried by the environment.")
		# prepare radii
		self._custom_radii = radii is not None # for the remarks of written surfaces
		table = radii if isinstance(radii, Mapping) else None # also for the environment
		if radii is None or isinstance(radii, Mapping):
			radii = atom_radii(ag, cg=cg, table=radii)
		else:
			radii = np.asarray(radii, dtype=float)
			if radii.shape != (ag.n_atoms,):
				raise ValueError(f"radii must have one value per atom ({ag.n_atoms}), got shape {radii.shape}.")

		self.cg = cg
		self.probe = probe
		self.bodies = bodies
		self.buried = buried
		self.save_atoms = save_atoms

		# prepare atoms/bodies: the only place that depends on the kind of bodies
		self.ag = ag
		self.radii = radii # (n_atoms,) van der Waals radii (nm)
		molecules, self.molecule_types, self.molecule_labels = partition(ag, bodies) # body of every atom, ...
		self.n_molecules = int(molecules.max()) + 1 if len(molecules) else 0
		self._molecule_of_atom = molecules
		self._members = molecule_members(molecules, self.n_molecules) # indices in ag of the atoms of every body
		# the bodies of every type (in order of appearance)
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
		self._body_anchor = anchor # (n_molecules,) of every body
		self._anchor = anchor[self._molecules] # (n_active,) of the body of every atom

		# environment: atoms with a radius that bury every body, see _copies
		self.environment = None
		if environment is not None:
			self._set_environment(environment, cg=cg, table=table)

		# parallelization
		self._max_frame_atoms = max_frame_atoms
		self._frames_per_block = max(1, max_frame_atoms // max(1, len(active)))

		# prep results of run(), see get_sasa / get_molecule_sasa
		self._lookup = np.full(ag.universe.atoms.n_atoms, -1, dtype=np.int64) # universe index -> index in ag
		self._lookup[ag.indices] = np.arange(ag.n_atoms)
		self.sasa = None # (n_atoms, n_frames) SASA (nm^2) of every atom of ag, only with save_atoms
		self.molecule_sasa = None # (n_molecules, n_frames) SASA (nm^2) of every molecule
		self.total_sasa = None # (n_frames,) total SASA (nm^2) of ag
		self.frames = None # (n_frames,) frame numbers
		self.times = None # (n_frames,) times (ps)
		self.free_sasa = self.free_molecule_sasa = self.free_total_sasa = None # with buried: without environment
		logger.debug("%d atoms (%d with a radius) in %s; %d dots per atom, probe %.3f nm; "
		             "%d frame(s) per parallel call", ag.n_atoms, len(active), self.describe(), len(self.unit),
		             probe, self._frames_per_block)

	def describe(self) -> str:
		"""The bodies in words, e.g. '3 molecule(s)', '12 residue(s)' or 'the whole selection as one body'."""
		if self.bodies == 'selection':
			text = 'the whole selection as one body'
		else:
			text = f"{self.n_molecules} {self.bodies}(s)"
		if self.environment is not None:
			text += f", covered by an environment of {self.environment.n_atoms} atom(s)"
		return text

	def _set_environment(self, environment: str | mda.AtomGroup, cg: bool, table: Mapping[str, float] | None) -> None:
		"""Check the environment and keep its atoms with a radius, their expanded radii and the cutoff."""
		universe = self.ag.universe
		if isinstance(environment, str):
			environment = universe.select_atoms(environment)
		elif hasattr(environment, 'atoms'):
			environment = environment.atoms
		else:
			raise TypeError(f"environment must be a selection string or an AtomGroup, got {type(environment).__name__}.")
		if environment.universe is not universe:
			raise ValueError("The environment belongs to a different Universe than the AtomGroup of SASA.")
		if not environment.n_atoms:
			raise ValueError("The environment contains no atoms.")
		overlap = np.intersect1d(environment.indices, self.ag.indices)
		if len(overlap):
			shown = ', '.join(atom_label(universe.atoms[i]) for i in overlap[:5])
			raise ValueError(f"{len(overlap)} atom(s) are both in the AtomGroup and in the environment: {shown}"
			                 f"{' ...' if len(overlap) > 5 else ''}; the environment must not contain them.")
		try:
			radii = atom_radii(environment, cg=cg, table=table)
		except ValueError as error:
			raise ValueError(f"environment: {error}") from None
		keep = radii > 0
		if not keep.any():
			raise ValueError("No atom of the environment has a radius.")
		self.environment = environment[keep]
		self._env_radii = radii[keep] # (n_env,) van der Waals radii (nm)
		self._env_ext = radii[keep] + self.probe # (n_env,) expanded radii (nm)
		# an environment atom can only bury an atom within the sum of their expanded radii
		self._env_cutoff = (self._ext.max() if len(self._ext) else 0.0) + self._env_ext.max()
		logger.debug("environment: %d atom(s) with a radius (of %d), cutoff %.3f nm", self.environment.n_atoms,
		             len(radii), self._env_cutoff)

	# SASA calculation
	def _atom_areas(self, xyz: np.ndarray, copies: list | None = None, coincident: list | None = None) -> np.ndarray:
		"""
		SASA (nm^2) of every atom in a block of frames.

		Parameters
		----------
		xyz : np.ndarray
			(n_frames, n_atoms, 3) positions (nm) of the atoms with a radius
		copies : list, optional
			of every frame the environment atoms near its bodies (see :meth:`_copies`),
			which bury them; None: without the environment
		coincident : list, optional
			gets (number of frames, (n_pairs, 2) universe indices) of the atoms at an
			identical position in this block of frames, if any

		Returns
		-------
		np.ndarray
			(n_frames, n_atoms)
		"""
		n_frames, n_atoms, _ = xyz.shape

		# put every (frame, molecule) block in its own slab along x and seperate them,
		# so a _neighbor_list finds the neighbours of all of them and no pairs between them
		anchors = xyz[:, self._body_anchor] # (n_frames, n_molecules, 3) position of every body
		xyz = (xyz - xyz[:, self._anchor]).reshape(-1, 3)
		block = (np.arange(n_frames)[:, None] * self.n_molecules + self._molecules[None]).ravel()
		ext = np.tile(self._ext, n_frames)
		n_target = len(xyz)
		target = env = None
		if copies is not None: # every copy in the block (slab) of its body, which only it buries
			env = np.concatenate([atoms for _, atoms, _ in copies])
			xyz = np.concatenate([xyz] + [positions - anchors[f, bodies]
			                              for f, (bodies, _, positions) in enumerate(copies)])
			block = np.concatenate([block] + [f * self.n_molecules + bodies for f, (bodies, _, _) in enumerate(copies)])
			ext = np.concatenate([ext, self._env_ext[env]])
			target = np.arange(len(xyz)) < n_target
		span = 2 * np.abs(xyz).max() + 2 * ext.max() + 0.1 # nm
		xyz[:, 0] += block * span

		exposed, pairs = _exposed_counts(xyz, ext, block, self.unit, target)
		if coincident is not None and len(pairs):
			indices = self._indices[pairs % n_atoms] # universe indices; the copies: of their environment atom
			if env is not None:
				copy = pairs >= n_target
				indices[copy] = self.environment.indices[env[pairs[copy] - n_target]]
			coincident.append((len(np.unique(pairs[:, 0] // n_atoms)), indices))
		return exposed[:n_target].reshape(n_frames, n_atoms) * self._dot_area

	def _copies(self, xyz: np.ndarray, ts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		The environment atoms near every body in one frame, in the periodic image of the
		body: (k,) body, (k,) index in :attr:`environment`, (k, 3) positions (nm); see
		:func:`environment_copies`.
		"""
		return environment_copies(xyz, self._molecules, to_nm(ts.positions[self.environment.indices]),
		                          self._env_cutoff, box_nm(ts.dimensions))

	def _position_blocks(self, start: int | None, stop: int | None, step: int | None,
	                     times: list | None = None) -> Iterator[tuple[np.ndarray, list | None]]:
		"""
		Positions (nm), (n_frames, n_atoms, 3), of the atoms with a radius in blocks of frames
		of ``start:stop:step``, with the environment copies of every frame (see :meth:`_copies`;
		None without an environment); the time of every frame is appended to ``times``.
		"""
		trajectory = self.ag.universe.trajectory
		current = trajectory.ts.frame
		buffer, copies, n_copies = [], [], 0
		with_environment = self.environment is not None
		try:
			for ts in trajectory[start:stop:step]:
				xyz = to_nm(ts.positions[self._indices]) # save positions (nm)
				buffer.append(xyz)
				if with_environment: # only the environment atoms within reach: never the whole environment
					copies.append(self._copies(xyz, ts))
					n_copies += len(copies[-1][0])
				if times is not None:
					times.append(ts.time)
				if (len(buffer) == self._frames_per_block or
				    n_copies and len(buffer) * len(self._indices) + n_copies >= self._max_frame_atoms):
					yield np.array(buffer), copies if with_environment else None # run SASA calculation
					buffer, copies, n_copies = [], [], 0
			if buffer:
				yield np.array(buffer), copies if with_environment else None
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
		:meth:`get_molecule_sasa`. With an environment, it is the SASA left uncovered by
		the environment; with ``buried`` also without the environment (:attr:`free_sasa`,
		..., see :meth:`get_buried_sasa`). A new run replaces the stored results.

		Parameters
		----------
		start, stop, step : int, optional
			frames of the trajectory, as in ``trajectory[start:stop:step]``
		per_atom : bool
			return the SASA of every atom instead of every molecule (needs ``save_atoms``).
			The SASA of an atom is that within its own body (and environment), so the
			atoms of a body sum to its SASA.
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
		def results() -> tuple[np.ndarray | None, np.ndarray]: # (atoms, molecules)
			return np.zeros((self.ag.n_atoms, n_frames)) if self.save_atoms else None, np.zeros((self.n_molecules, n_frames))

		def store(areas: np.ndarray, into: tuple, block: slice) -> None:
			atoms, molecules = into
			if atoms is not None:
				atoms[self._active, block] = areas.T
			molecules[:, block] = self._molecule_areas(areas).T

		shielded = results()
		free = results() if self.buried else None
		times, coincident = [], []
		n_copies = 0
		tic = time.perf_counter()
		if len(self._indices) and n_frames: # else only atoms without radius / no frames
			done = 0
			with ProgressBar(total=n_frames, disable=not verbose, desc=f"SASA calculation {name}", unit="frame") as bar:
				for xyz, copies in self._position_blocks(start, stop, step, times):
					block = slice(done, done + len(xyz))
					store(self._atom_areas(xyz, copies, coincident), shielded, block)
					if free is not None: # the same positions without the environment
						store(self._atom_areas(xyz), free, block)
					n_copies += sum(len(bodies) for bodies, _, _ in copies or [])
					done += len(xyz)
					bar.update(len(xyz))
		else:
			current = trajectory.ts.frame
			times = [ts.time for ts in trajectory[start:stop:step]]
			trajectory[current]
		if coincident: # once per run, not per block of frames
			universe = self.ag.universe
			warn_coincident(np.concatenate([pairs for _, pairs in coincident]),
			                label=lambda i: atom_label(universe.atoms[i]), frames=sum(n for n, _ in coincident))
		# results only replaced after a complete run
		self.frames, self.times = frames, np.array(times, dtype=float)
		self.sasa, self.molecule_sasa = shielded
		self.total_sasa = self.molecule_sasa.sum(axis=0)
		if free is not None:
			self.free_sasa, self.free_molecule_sasa = free
			self.free_total_sasa = self.free_molecule_sasa.sum(axis=0)
		else:
			self.free_sasa = self.free_molecule_sasa = self.free_total_sasa = None
		elapsed = time.perf_counter() - tic
		logger.debug("SASA of %d frame(s) in %.2f s (%.1f frames/s)", n_frames, elapsed,
		             n_frames / elapsed if elapsed > 0 else float('inf'))
		if self.environment is not None and n_frames and self.n_molecules:
			logger.debug("environment: %.1f atom(s) within reach per %s and frame", n_copies / n_frames / self.n_molecules,
			             'body' if self.bodies == 'selection' else self.bodies)
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

	def _check_run(self, atoms: bool = False, buried: bool = False) -> None:
		'''
		Check that a SASA has been computed with SASA.run, that the SASA of every atom is available if ``atoms``,
		and the SASA without the environment if ``buried``. Raises a RuntimeError otherwise.
		'''
		if self.frames is None:
			raise RuntimeError("No SASA computed yet: call run() first.")
		if atoms and self.sasa is None:
			raise RuntimeError("The SASA of every atom was not saved (save_atoms=False): only the total "
			                   "and molecule SASA are available.")
		if buried and self.free_molecule_sasa is None:
			raise RuntimeError("The buried SASA needs an environment and buried=True.")

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

	def get_buried_sasa(self, selection: str | mda.AtomGroup | None = None, per_atom: bool = False) -> np.ndarray:
		"""
		Area (nm^2) over time of a selection of the atoms of ag that the environment buries:
		its SASA without minus with the environment, from the last SASA.run(). Needs an
		environment and ``buried=True``; arguments and shapes as :meth:`get_sasa`.
		"""
		if selection is None and not per_atom:
			self._check_run(buried=True)
			return self.free_total_sasa - self.total_sasa
		self._check_run(atoms=True, buried=True)
		indices = self._ag_indices(selection)
		atoms = self.free_sasa[indices] - self.sasa[indices]
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
			whole = " (the whole selection is one body, index 0)" if self.bodies == 'selection' else ""
			raise IndexError(f"Molecule index {molecule} out of range for {self.n_molecules} molecule(s){whole}.")
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
		return self._molecule_rows(self.molecule_sasa, index)

	def get_buried_molecule_sasa(self, index: int | str | ArrayLike | None = None) -> np.ndarray:
		"""
		Area (nm^2) over time of molecules that the environment buries: their SASA without
		minus with the environment, from the last SASA.run(). Needs an environment and
		``buried=True``; arguments and shapes as :meth:`get_molecule_sasa`.
		"""
		self._check_run(buried=True)
		return self._molecule_rows(self.free_molecule_sasa - self.molecule_sasa, index)

	def _molecule_rows(self, values: np.ndarray, index: int | str | ArrayLike | None) -> np.ndarray:
		"""Rows of (n_molecules, n_frames) ``values`` by molecule index, indices or label, see :meth:`get_molecule_sasa`."""
		if index is None:
			return values
		if isinstance(index, str):
			return values[self.molecule_number(index)]
		index = np.asarray(index)
		if not np.issubdtype(index.dtype, np.integer):
			raise TypeError(f"Molecule indices must be integers, got {index.dtype}.")
		if ((index < -self.n_molecules) | (index >= self.n_molecules)).any():
			raise IndexError(f"Molecule index out of range for {self.n_molecules} molecule(s): {index.tolist()}.")
		return values[index]


	# SASA surface visualization
	def molecule_atoms(self, molecule: int | str | None = None) -> mda.AtomGroup:
		"""
		Atoms of a molecule (index as in :meth:`get_molecule_sasa`, negative counts from
		the end, or label); all atoms of ``ag`` for None.
		"""
		if molecule is None:
			return self.ag
		return self.ag[self._members[self.molecule_number(molecule)]]

	def _molecule_remark(self, atoms: mda.AtomGroup, molecule: int) -> str:
		"""REMARK record naming the atoms of a written surface: which molecule, or the whole AtomGroup."""
		name = atoms.moltypes[0] if hasattr(atoms, 'moltypes') else (
		       '/'.join(dict.fromkeys(atoms.resnames)) if hasattr(atoms, 'resnames') else 'molecule')
		resids = atoms.resids if hasattr(atoms, 'resids') else atoms.resindices + 1
		residues = f"resid {resids.min()}" if resids.min() == resids.max() else f"resids {resids.min()}-{resids.max()}"
		described = f"{name}, {residues}, {atoms.n_atoms} atoms"
		if self.bodies == 'selection':
			text = f"Whole selection as one body ({described})"
		else:
			text = f"{self.bodies.capitalize()} {molecule % self.n_molecules} of {self.n_molecules} ({described}), on its own"
		if self.environment is not None:
			text += f"; covered by an environment of {self.environment.n_atoms} atoms"
		return text

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
			fixed trajectory frame (negative counts from the end): only the molecule is chosen
			(the first of the type if the frame was not analysed); IndexError if out of range

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
		n_frames = self.ag.universe.trajectory.n_frames
		if not -n_frames <= frame < n_frames:
			raise IndexError(f"Frame {frame} out of range for {n_frames} frame(s).")
		frame = int(frame) % n_frames
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

		The surface is that of one body (see ``bodies``): a molecule or residue on its own,
		or the whole AtomGroup with ``bodies='selection'``. With an environment, only the
		part left uncovered by it; the environment atoms within reach are written too
		(chain B).

		Statistics are taken over the SASA distribution of the body's type only (all
		bodies of the type in all analysed frames; with ``bodies='selection'``: the total),
		so every type has its own median, mean, ...

		Parameters
		----------
		fname : str
			PDB file to write
		frame : int, optional
			trajectory frame (negative counts from the end); default the analysed frame of the
			molecule whose SASA is closest to ``statistic`` (by default the median) of its
			type's distribution, i.e. a representative structure
		molecule : int or str
			index of the body (as in :meth:`get_molecule_sasa`, negative counts from the
			end) or its label (e.g. 'GSH2'), default the first; with
			``bodies='selection'`` the whole AtomGroup is the only body (0, 'selection')
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
			molecule = self.molecule_number(molecule)
			if frame is None and reference is None:
				self._check_run()
				target = float(function(self.get_type_sasa(self.molecule_types[molecule]).ravel()))
				series = self.get_molecule_sasa(molecule)
				frame = int(self.frames[np.argmin(np.abs(series - target))])
				chosen = self._chosen_remark(molecule, frame, target, name)
		atoms = self.molecule_atoms(molecule) # bodies='selection': all of ag
		indices = self._lookup[atoms.indices]
		environment = {} if self.environment is None else {'environment': self.environment,
		                                                   'environment_radii': self._env_radii}
		result = generate_surface(atoms, radii=self.radii[indices], custom_radii=self._custom_radii,
		                          probe=self.probe, block=self._molecule_of_atom[indices], frame=frame,
		                          reference=reference, frames=self.frames, ndots=ndots, cg=self.cg, **environment)
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
		pool = (f"{len(self.frames)} analysed frame(s)" if self.bodies == 'selection' else
		        f"{n_molecules} {molecule_type} {self.bodies}(s) x {len(self.frames)} analysed frame(s)")
		what = (f"Representative molecule and frame: {self.molecule_labels[molecule]} in frame {frame}"
		        if molecule_choice else "Representative frame")
		logger.info("%s %s in frame %d: SASA %s, closest to the %s (%.3f nm^2) of %s",
		            'representative' if molecule_choice else 'surface of', self.molecule_labels[molecule], frame,
		            sasa, name, target, pool)
		return f"{what}: SASA ({sasa}) closest to the {name} ({target:.3f} nm^2) of {pool}"
