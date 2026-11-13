"""
SASA of every body (molecule, residue, or the whole selection) and of every atom of an
MDAnalysis AtomGroup over a trajectory, optionally covered by an environment.
"""
from collections.abc import Callable, Iterator, Mapping
import logging
from pathlib import Path
import time
import numpy as np
from numpy.typing import ArrayLike
import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar

from .dots import unit_dots
from .engine import _exposed_counts, warn_coincident
from .environment import box_nm, environment_copies
from .groups import BODIES, Bodies, atom_label, members, partition, body_type_indices
from .radii import PROBE_RADIUS, atom_radii
from .results import Results, _difference, body_index, save
from .units import to_nm
from .surface import SURFACE_DOTS, Surface, _statistic, generate_surface

logger = logging.getLogger(__name__)

N_DOTS = 4800 # default minimum number of dots per atom (-> 4842 dots)


class SASA:
	"""
	Solvent accessible surface area of every body of an MDAnalysis AtomGroup
	(Shrake-Rupley) over a trajectory, and of every atom.

	"Atom" means every particle of the AtomGroup, as in MDAnalysis: an atom in
	atomistic systems, a coarse-grained bead with ``cg=True``.

	Two independent settings decide which atoms bury the surface of an atom:

	- ``bodies``: the AtomGroup is split into bodies, and only atoms of the same body
	  bury each other. By default every molecule is a body; 'residue' makes every residue a 
	  body, 'selection' the whole AtomGroup (all its atoms bury each other).
	- ``environment``: atoms outside the AtomGroup (e.g. a protein around a ligand)
	  that bury every body, but whose own SASA is not computed. Every body is covered
	  by the environment, not by the other bodies (unless they are one body).

	Atoms with radius 0 (dummy beads, virtual sites) are ignored. Bodies are assumed
	to be whole.

	:meth:`run` returns the :class:`Results` (also :attr:`results`): the SASA of every
	body, in total, by type, of every atom and of any selection of atoms.

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
		also compute the SASA without the environment (:attr:`free`), and so the area the
		environment buries (:attr:`buried`); needs an ``environment``
	keep_atoms : bool
		keep the SASA of every atom of every frame (``results.atom_sasa``, (n_atoms, n_frames)),
		needed for :meth:`Results.select`. The SASA of every body is always kept; False
		saves the memory of the per-atom array for very large systems or long trajectories

	Examples
	--------
	>>> sasa = SASA(universe.select_atoms('moltype GSH'))
	>>> res = sasa.run()          # Results, also sasa.results
	>>> res.body_sasa, res.body_labels # (n_bodies, n_frames) nm^2, ['GSH1', 'GSH2', ...]
	>>> res.total_sasa            # (n_frames,)
	>>> res.select('name S*')     # (n_frames,) SASA of the sulfur atoms
	>>> sasa.write_surface('dots.pdb', body='GSH1')  # structure + surface dots of a median-SASA frame
	>>> ligand = SASA(universe.select_atoms('resname LIG'), environment='protein', buried=True)
	>>> ligand.run().total_sasa   # SASA of the ligand left uncovered by the protein
	>>> ligand.buried.total_sasa  # area of the ligand buried by the protein
	>>> ligand.save('ligand.npz') # all of it, see load()

	Attributes
	----------
	bodies : str
		how the AtomGroup is split into bodies ('molecule', 'residue' or 'selection')
	environment : MDAnalysis.AtomGroup or None
		the environment atoms with a radius, None without an environment
	body_labels : list of str
		(n_bodies,) label of every body: its type numbered within the type (GSH1, GSH2,
		SOL1, ...); ``<resname><resid>`` (ALA12) with ``bodies='residue'``; 'selection'
		with ``bodies='selection'``
	body_type_labels : list of str
		(n_bodies,) type of every body: its moltype (tpr), else named by its first residue
		(see :func:`molecule_types`); the residue type (see :func:`residue_types`) with
		``bodies='residue'``; 'selection' with ``bodies='selection'``
	body_type_indices : dict of str to np.ndarray
		body indices of every type, in the order of first appearance
	results : Results or None
		of the last :meth:`run`: the SASA (with an environment: the part it leaves uncovered)
	free, buried : Results or None
		of the last :meth:`run` with ``buried``: the SASA without the environment, and the
		area the environment buries (free minus results); None otherwise
	"""
	
	def __init__(self, ag: mda.AtomGroup, cg: bool = False, probe: float = PROBE_RADIUS, ndots: int = N_DOTS,
	             radii: ArrayLike | Mapping[str, float] | None = None, max_frame_atoms: int = 20000,
	             bodies: Bodies = 'molecule', environment: str | mda.AtomGroup | None = None,
	             keep_atoms: bool = True, buried: bool = False) -> None:

		# check parameters
		if bodies not in BODIES:
			raise ValueError(f"bodies must be one of {', '.join(BODIES)}, got {bodies!r}.")
		if not probe >= 0:
			raise ValueError(f"The probe radius must be >= 0 (nm), got {probe}.")
		if buried and environment is None:
			raise ValueError("buried needs an environment: the area buried by the environment.")
		
		# prepare radii
		self._custom_radii = radii is not None # for the remarks of written surfaces
		radii_table = radii if isinstance(radii, Mapping) else None # also for the environment
		if radii is None or isinstance(radii, Mapping):
			radii = atom_radii(ag, cg=cg, table=radii)
		else:
			radii = np.asarray(radii, dtype=float)
			if radii.shape != (ag.n_atoms,):
				raise ValueError(f"radii must have one value per atom ({ag.n_atoms}), got shape {radii.shape}.")

		# store parameters
		self.cg = cg
		self.probe = probe
		self.bodies = bodies
		self.keep_atoms = keep_atoms
		self._with_buried = buried

		# prepare atoms/bodies
		self.ag = ag
		self.radii = radii # (n_atoms,) van der Waals radii (nm)
		body_of_atom, self.body_type_labels, self.body_labels = partition(ag, bodies)
		self.n_bodies = int(body_of_atom.max()) + 1 if len(body_of_atom) else 0
		self._body_of_atom = body_of_atom # (n_atoms,) body of every atom
		self._members = members(body_of_atom, self.n_bodies) # indices in ag of the atoms of every body
		active = np.flatnonzero(radii > 0) # select only atoms with a radius
		self._active = active # indices in ag of the atoms with a radius
		self._indices = ag.indices[active]
		self._body = body_of_atom[active] # (n_active,) body of every atom with a radius

		# prepare dots
		self.unit = unit_dots(int(ndots))
		self._ext = radii[active] + probe # expanded radii (nm)
		self._dot_area = 4 * np.pi * self._ext**2 / len(self.unit) # nm^2 per dot

		# every body is placed relative to its first atom
		# anchor is the index of the first atom of every body in the active atoms
		_, first = np.unique(self._body, return_index=True)
		anchor = np.zeros(self.n_bodies, dtype=int)
		anchor[self._body[first]] = first
		self._body_anchor = anchor # (n_bodies,) of every body
		self._anchor = anchor[self._body] # (n_active,) of the body of every atom

		# environment: atoms with a radius that bury every body, see _copies
		self.environment = None
		if environment is not None:
			self._set_environment(environment, cg=cg, radii_table=radii_table)

		# parallelization
		self._max_frame_atoms = max_frame_atoms
		self._frames_per_block = max(1, max_frame_atoms // max(1, len(active)))

		self._lookup = np.full(ag.universe.atoms.n_atoms, -1, dtype=np.int64) # universe index -> index in ag
		self._lookup[ag.indices] = np.arange(ag.n_atoms)
		self.results = self.free = self.buried = None # of run()
		logger.debug("%d atoms (%d with a radius) in %s; %d dots per atom, probe %.3f nm; "
		             "%d frame(s) per parallel call", ag.n_atoms, len(active), self.describe(), len(self.unit),
		             probe, self._frames_per_block)

	@property
	def body_type_indices(self) -> dict[str, np.ndarray]:
		"""Body indices of every type, in the order of first appearance: {type: indices}."""
		return body_type_indices(self.body_type_labels)

	def describe(self) -> str:
		"""The bodies in words, e.g. '3 molecule(s)', '12 residue(s)' or 'the whole selection as one body'."""
		if self.bodies == 'selection':
			text = 'the whole selection as one body'
		else:
			text = f"{self.n_bodies} {self.bodies}(s)"
		if self.environment is not None:
			text += f", covered by an environment of {self.environment.n_atoms} atom(s)"
		return text

	def _set_environment(self, environment: str | mda.AtomGroup, cg: bool, radii_table: Mapping[str, float] | None) -> None:
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
			radii = atom_radii(environment, cg=cg, table=radii_table)
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

		# put every (frame, body) block in its own slab along x and seperate them,
		# so a _neighbor_list finds the neighbours of all of them and no pairs between them
		anchors = xyz[:, self._body_anchor] # (n_frames, n_bodies, 3) position of every body
		xyz = (xyz - xyz[:, self._anchor]).reshape(-1, 3)
		block = (np.arange(n_frames)[:, None] * self.n_bodies + self._body[None]).ravel()
		ext = np.tile(self._ext, n_frames)
		n_target = len(xyz)
		target = env = None
		if copies is not None: # every copy in the block (slab) of its body, which only it buries
			env = np.concatenate([atoms for _, atoms, _ in copies])
			xyz = np.concatenate([xyz] + [positions - anchors[f, bodies]
			                              for f, (bodies, _, positions) in enumerate(copies)])
			block = np.concatenate([block] + [f * self.n_bodies + bodies for f, (bodies, _, _) in enumerate(copies)])
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
		return environment_copies(xyz, self._body, to_nm(ts.positions[self.environment.indices]),
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

	def _body_areas(self, areas: np.ndarray) -> np.ndarray:
		"""(n_frames, n_atoms) atom SASA -> (n_frames, n_bodies) body SASA."""
		n_frames = len(areas)
		block = (np.arange(n_frames)[:, None] * self.n_bodies + self._body[None]).ravel()
		return np.bincount(block, weights=areas.ravel(),
		                   minlength=n_frames * self.n_bodies).reshape(n_frames, -1)

	def run(self, start: int | None = None, stop: int | None = None, step: int | None = None,
	        verbose: bool = False, name: str = "") -> Results:
		"""
		SASA (nm^2) of every body, and with ``keep_atoms`` of every atom, in the frames
		``start:stop:step`` of the trajectory.

		With an environment, it is the SASA left uncovered by the environment; with
		``buried`` also without the environment (:attr:`free`) and the area the environment
		buries (:attr:`buried`). A new run replaces the stored results.

		Parameters
		----------
		start, stop, step : int, optional
			frames of the trajectory, as in ``trajectory[start:stop:step]``
		verbose : bool
			show a progress bar over the frames. It advances once per block of frames
			(see ``max_frame_atoms``), so it costs no measurable time but moves in steps.
		name : str
			label of the progress bar

		Returns
		-------
		Results
		"""
		trajectory = self.ag.universe.trajectory
		frames = np.arange(trajectory.n_frames)[start:stop:step]
		n_frames = len(frames)

		def arrays() -> tuple[np.ndarray | None, np.ndarray]: # (atoms, bodies)
			return np.zeros((self.ag.n_atoms, n_frames)) if self.keep_atoms else None, np.zeros((self.n_bodies, n_frames))

		def store(areas: np.ndarray, into: tuple, block: slice) -> None:
			atoms, bodies = into
			if atoms is not None:
				atoms[self._active, block] = areas.T
			bodies[:, block] = self._body_areas(areas).T

		covered = arrays()
		free = arrays() if self._with_buried else None
		times, coincident = [], []
		n_copies = 0
		tic = time.perf_counter()
		if len(self._indices) and n_frames: # else only atoms without radius / no frames
			done = 0
			with ProgressBar(total=n_frames, disable=not verbose, desc=f"SASA calculation {name}", unit="frame") as bar:
				for xyz, copies in self._position_blocks(start, stop, step, times):
					block = slice(done, done + len(xyz))
					store(self._atom_areas(xyz, copies, coincident), covered, block)
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
		atom_labels = [atom_label(atom) for atom in self.ag] if self.keep_atoms else None

		def results(atoms: np.ndarray | None, bodies: np.ndarray) -> Results:
			return Results(times, frames, bodies, self.body_labels, self.body_type_labels, atoms, atom_labels, self.ag)

		self.results = results(*covered)
		self.free = results(*free) if free is not None else None
		self.buried = _difference(self.free, self.results) if free is not None else None
		elapsed = time.perf_counter() - tic
		logger.debug("SASA of %d frame(s) in %.2f s (%.1f frames/s)", n_frames, elapsed,
		             n_frames / elapsed if elapsed > 0 else float('inf'))
		if self.environment is not None and n_frames and self.n_bodies:
			logger.debug("environment: %.1f atom(s) within reach per %s and frame", n_copies / n_frames / self.n_bodies,
			             'body' if self.bodies == 'selection' else self.bodies)
		return self.results

	def _check_run(self) -> Results:
		"""The results of the last run; RuntimeError before a run."""
		if self.results is None:
			raise RuntimeError("No SASA computed yet: call run() first.")
		return self.results

	def save(self, fname: str | Path, table: tuple[list[str], np.ndarray] | None = None, **meta) -> None:
		"""
		Save the results of the last :meth:`run` (with ``buried`` also without the
		environment) and how they were computed to an npz file, see :func:`load`.

		Parameters
		----------
		fname : str or Path
			npz file
		table : tuple, optional
			(labels, (n_columns, n_frames) values) of a table to save as well (e.g. from
			:meth:`Results.table`)
		**meta
			more information to keep (e.g. the selection), JSON-serialisable
		"""
		from . import __version__
		results = self._check_run()
		info = {'sassy': __version__, 'bodies': self.bodies, 'probe': self.probe, 'ndots': len(self.unit),
		        'cg': self.cg, 'n_atoms': self.ag.n_atoms,
		        'n_environment': None if self.environment is None else self.environment.n_atoms}
		save(fname, results, self.free, meta=info | meta, table=table)

	# bodies
	def body_index(self, body: int | str) -> int:
		"""Index (0, 1, ...) of a body by its index (negative counts from the end) or label (e.g. 'GSH2')."""
		return body_index(self.body_labels, body)

	def body_atoms(self, body: int | str | None = None) -> mda.AtomGroup:
		"""Atoms of a body (index, negative counts from the end, or label); all atoms of ``ag`` for None."""
		if body is None:
			return self.ag
		return self.ag[self._members[self.body_index(body)]]

	# SASA surface visualization
	def _body_remark(self, atoms: mda.AtomGroup, body: int) -> str:
		"""REMARK record naming the atoms of a written surface: which body, or the whole AtomGroup."""
		name = atoms.moltypes[0] if hasattr(atoms, 'moltypes') else (
		       '/'.join(dict.fromkeys(atoms.resnames)) if hasattr(atoms, 'resnames') else 'molecule')
		resids = atoms.resids if hasattr(atoms, 'resids') else atoms.resindices + 1
		residues = f"resid {resids.min()}" if resids.min() == resids.max() else f"resids {resids.min()}-{resids.max()}"
		described = f"{name}, {residues}, {atoms.n_atoms} atoms"
		if self.bodies == 'selection':
			text = f"Whole selection as one body ({described})"
		else:
			text = f"{self.bodies.capitalize()} {body % self.n_bodies} of {self.n_bodies} ({described}), on its own"
		if self.environment is not None:
			text += f"; covered by an environment of {self.environment.n_atoms} atoms"
		return text

	def representative(self, body_type: str, statistic: str | Callable[[np.ndarray], float] | float = 'median',
	                   frame: int | None = None) -> tuple[int, int, float]:
		"""
		A representative body of a type and frame: the statistic is taken over the SASA
		distribution of that type only (all its bodies in all analysed frames of the last
		:meth:`run`), and the (body, frame) whose SASA is closest to it is chosen.

		Parameters
		----------
		body_type : str
			type (see :attr:`types`)
		statistic : str, callable or float
			see :meth:`write_surface`
		frame : int, optional
			fixed trajectory frame (negative counts from the end): only the body is chosen
			(the first of the type if the frame was not analysed); IndexError if out of range

		Returns
		-------
		body : int
			body index
		frame : int
			trajectory frame
		target : float
			the statistic (nm^2) of the type's distribution
		"""
		results = self._check_run()
		if body_type not in self.body_type_indices:
			raise KeyError(f"No body type {body_type!r}; types: {', '.join(self.body_type_indices)}.")
		function, _ = _statistic(statistic)
		bodies = self.body_type_indices[body_type]
		values = results.body_sasa[bodies] # (n_bodies_of_type, n_frames)
		target = float(function(values.ravel()))
		if frame is None:
			k, column = np.unravel_index(np.argmin(np.abs(values - target)), values.shape)
			return int(bodies[k]), int(results.frames[column]), target
		n_frames = self.ag.universe.trajectory.n_frames
		if not -n_frames <= frame < n_frames:
			raise IndexError(f"Frame {frame} out of range for {n_frames} frame(s).")
		frame = int(frame) % n_frames
		analysed = np.flatnonzero(results.frames == frame)
		if not len(analysed):
			return int(bodies[0]), frame, target
		return int(bodies[np.argmin(np.abs(values[:, analysed[0]] - target))]), frame, target

	def write_surface(self, fname: str, frame: int | None = None, body: int | str = 0,
	                  reference: ArrayLike | mda.AtomGroup | None = None,
	                  ndots: int = SURFACE_DOTS, return_surface: bool = False,
	                  statistic: str | Callable[[np.ndarray], float] | float = 'median',
	                  body_type: str | None = None) -> Surface | None:
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
			body whose SASA is closest to ``statistic`` (by default the median) of its
			type's distribution, i.e. a representative structure
		body : int or str
			index of the body (negative counts from the end) or its label (e.g. 'GSH2'),
			default the first; with ``bodies='selection'`` the whole AtomGroup is the only
			body (0, 'selection')
		body_type : str, optional
			instead of ``body``: a type (see :attr:`types`); its representative body and
			frame are used, see :meth:`representative` (with ``frame``: the body closest to
			the statistic in that frame; with ``reference``: its first body)
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
		if body_type is not None:
			if body_type not in self.body_type_indices:
				raise KeyError(f"No body type {body_type!r}; types: {', '.join(self.body_type_indices)}.")
			if reference is not None:
				body = int(self.body_type_indices[body_type][0])
			else:
				body, frame, target = self.representative(body_type, function, frame)
				chosen = self._chosen_remark(body, frame, target, name, body_choice=True)
		else:
			body = self.body_index(body)
			if frame is None and reference is None:
				results = self._check_run()
				target = float(function(results.type_sasa(per_body=True)[self.body_type_labels[body]].ravel()))
				frame = int(results.frames[np.argmin(np.abs(results.body_sasa[body] - target))])
				chosen = self._chosen_remark(body, frame, target, name)
		atoms = self.body_atoms(body) # bodies='selection': all of ag
		indices = self._lookup[atoms.indices]
		environment = {} if self.environment is None else {'environment': self.environment,
		                                                   'environment_radii': self._env_radii}
		result = generate_surface(atoms, radii=self.radii[indices], custom_radii=self._custom_radii,
		                          probe=self.probe, block=self._body_of_atom[indices], frame=frame,
		                          reference=reference, frames=None if self.results is None else self.results.frames,
		                          ndots=ndots, cg=self.cg, **environment)
		result.remarks.insert(2, self._body_remark(atoms, body)) # after the frame
		if chosen is not None:
			result.remarks.insert(3, chosen)
		result.body = body
		result.write(fname)
		logger.info("wrote the surface of %s in frame %d (%d atoms, %d dots, SASA %.3f nm^2) to %s",
		            self.body_labels[body], result.frame, result.atoms.n_atoms, len(result.dots), result.area, fname)
		return result if return_surface else None

	def _chosen_remark(self, body: int, frame: int, target: float, name: str, body_choice: bool = False) -> str:
		"""REMARK on how the frame (and body) of a surface was chosen; also logged."""
		results = self.results
		body_type = self.body_type_labels[body]
		n_bodies = len(self.body_type_indices[body_type])
		analysed = np.flatnonzero(results.frames == frame)
		sasa = f"{results.body_sasa[body, analysed[0]]:.3f} nm^2" if len(analysed) else "not analysed"
		pool = (f"{results.n_frames} analysed frame(s)" if self.bodies == 'selection' else
		        f"{n_bodies} {body_type} {self.bodies}(s) x {results.n_frames} analysed frame(s)")
		what = (f"Representative {self.bodies} and frame: {self.body_labels[body]} in frame {frame}"
		        if body_choice else "Representative frame")
		logger.info("%s %s in frame %d: SASA %s, closest to the %s (%.3f nm^2) of %s",
		            'representative' if body_choice else 'surface of', self.body_labels[body], frame,
		            sasa, name, target, pool)
		return f"{what}: SASA ({sasa}) closest to the {name} ({target:.3f} nm^2) of {pool}"
