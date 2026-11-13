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
from .environment import Copies, box_nm, environment_copies
from .groups import BODIES, Bodies, atom_label, body_type_indices, members, partition
from .radii import PROBE_RADIUS, atom_radii, radii_remarks, resolve_radii
from .results import Results, _difference, body_index, save
from .statistics import resolve_statistic
from .surface import SURFACE_DOTS, Surface, body_remark, chosen_remark, generate_surface
from .units import to_nm

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
	keep_atoms : bool
		keep the SASA of every atom of every frame (``results.atom_sasa``, (n_atoms, n_frames)),
		needed for :meth:`Results.select`. The SASA of every body is always kept; False
		saves the memory of the per-atom array for very large systems or long trajectories
	buried : bool
		also compute the SASA without the environment (:attr:`free`), and so the area the
		environment buries (:attr:`buried`); needs an ``environment``

	Examples
	--------
	>>> sasa = SASA(universe.select_atoms('moltype GSH'))
	>>> res = sasa.run()                # Results, also sasa.results
	>>> res.body_sasa, res.body_labels  # (n_bodies, n_frames) nm^2, ['GSH1', 'GSH2', ...]
	>>> res.total_sasa                  # (n_frames,)
	>>> res.select('name S*')           # (n_frames,) SASA of the sulfur atoms
	>>> sasa.surface('GSH1').write('dots.pdb')  # structure + surface dots of a median-SASA frame
	>>> ligand = SASA(universe.select_atoms('resname LIG'), environment='protein', buried=True)
	>>> ligand.run().total_sasa         # SASA of the ligand left uncovered by the protein
	>>> ligand.buried.total_sasa        # area of the ligand buried by the protein
	>>> ligand.save('ligand.npz')       # all of it, see load()

	Attributes
	----------
	bodies : str
		how the AtomGroup is split into bodies ('molecule', 'residue' or 'selection')
	n_bodies : int
		number of bodies
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
	environment : MDAnalysis.AtomGroup or None
		the environment atoms with a radius, None without an environment
	radii : np.ndarray
		(n_atoms,) van der Waals radii (nm) of the atoms of ``ag``
	dots : np.ndarray
		(n_dots, 3) dots on the unit sphere, scaled onto every atom
	results : Results or None
		of the last :meth:`run`: the SASA (with an environment: the part it leaves uncovered)
	free, buried : Results or None
		of the last :meth:`run` with ``buried``: the SASA without the environment, and the
		area the environment buries (free minus results); None otherwise
	"""

	# --- setup
	def __init__(self, ag: mda.AtomGroup, cg: bool = False, probe: float = PROBE_RADIUS, ndots: int = N_DOTS,
	             radii: ArrayLike | Mapping[str, float] | None = None, max_frame_atoms: int = 20000,
	             bodies: Bodies = 'molecule', environment: str | mda.AtomGroup | None = None,
	             keep_atoms: bool = True, buried: bool = False) -> None:
		if bodies not in BODIES:
			raise ValueError(f"bodies must be one of {', '.join(BODIES)}, got {bodies!r}.")
		if not probe >= 0:
			raise ValueError(f"The probe radius must be >= 0 (nm), got {probe}.")
		if buried and environment is None:
			raise ValueError("buried needs an environment: the area buried by the environment.")
		self.ag, self.cg, self.probe, self.bodies, self.keep_atoms = ag, cg, probe, bodies, keep_atoms
		self._compute_free = buried # also without the environment, see run()
		self._custom_radii = radii is not None # for the remarks of written surfaces
		self.radii, radii_table = resolve_radii(ag, radii, cg)
		self.dots = unit_dots(int(ndots))
		self._prepare_bodies()
		self.environment = None
		if environment is not None:
			self._set_environment(environment, radii_table)
		self._max_frame_atoms = max_frame_atoms
		self._frames_per_block = max(1, max_frame_atoms // max(1, len(self._active)))
		self.results = self.free = self.buried = None # of run()
		logger.debug("%d atoms (%d with a radius) in %s; %d dots per atom, probe %.3f nm; %d frame(s) per "
		             "parallel call", ag.n_atoms, len(self._active), self, len(self.dots), probe, self._frames_per_block)

	def _prepare_bodies(self) -> None:
		"""Split ``ag`` into bodies, and the arrays of its atoms with a radius (the active atoms)."""
		ag = self.ag
		self._body_of_atom, self.body_type_labels, self.body_labels = partition(ag, self.bodies)
		self._members = members(self._body_of_atom, self.n_bodies) # positions in ag of the atoms of every body
		self._ag_position = np.full(ag.universe.atoms.n_atoms, -1, dtype=np.int64) # universe index -> position in ag
		self._ag_position[ag.indices] = np.arange(ag.n_atoms)

		# the active atoms: with a radius; atoms with radius 0 neither get nor bury dots
		active = np.flatnonzero(self.radii > 0)
		self._active = active # positions in ag
		self._active_universe = ag.indices[active] # universe indices
		self._active_body = self._body_of_atom[active]
		self._active_ext = self.radii[active] + self.probe # expanded radii (nm)
		self._dot_area = 4 * np.pi * self._active_ext**2 / len(self.dots) # nm^2 per dot
		# every body is placed relative to its first active atom (see _atom_areas)
		_, first = np.unique(self._active_body, return_index=True)
		self._body_anchor = np.zeros(self.n_bodies, dtype=int) # (n_bodies,) position in the active atoms
		self._body_anchor[self._active_body[first]] = first
		self._active_anchor = self._body_anchor[self._active_body]

	def _set_environment(self, environment: str | mda.AtomGroup, radii_table: Mapping[str, float] | None) -> None:
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
			radii = atom_radii(environment, cg=self.cg, table=radii_table)
		except ValueError as error:
			raise ValueError(f"environment: {error}") from None
		keep = radii > 0
		if not keep.any():
			raise ValueError("No atom of the environment has a radius.")
		self.environment = environment[keep]
		self._env_radii = radii[keep] # (n_env,) van der Waals radii (nm)
		self._env_ext = radii[keep] + self.probe # (n_env,) expanded radii (nm)
		# an environment atom can only bury an atom within the sum of their expanded radii
		self._env_cutoff = (self._active_ext.max() if len(self._active_ext) else 0.0) + self._env_ext.max()
		logger.debug("environment: %d atom(s) with a radius (of %d), cutoff %.3f nm", self.environment.n_atoms,
		             len(radii), self._env_cutoff)

	@property
	def n_bodies(self) -> int:
		return len(self.body_labels)

	@property
	def body_type_indices(self) -> dict[str, np.ndarray]:
		"""Body indices of every type, in the order of first appearance: {type: indices}."""
		return body_type_indices(self.body_type_labels)

	def __str__(self) -> str:
		"""The bodies in words, e.g. '3 molecule(s)', '12 residue(s)' or 'the whole selection as one body'."""
		text = 'the whole selection as one body' if self.bodies == 'selection' else f"{self.n_bodies} {self.bodies}(s)"
		if self.environment is not None:
			text += f", covered by an environment of {self.environment.n_atoms} atom(s)"
		return text

	def __repr__(self) -> str:
		return f"<SASA: {self.ag.n_atoms} atoms in {self}>"

	# --- running
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
		frames = np.arange(self.ag.universe.trajectory.n_frames)[start:stop:step]
		covered = self._new_arrays(len(frames))
		free = self._new_arrays(len(frames)) if self._compute_free else None
		times, coincident, done = [], [], 0
		tic = time.perf_counter()
		with ProgressBar(total=len(frames), disable=not verbose, desc=f"SASA calculation {name}", unit="frame") as bar:
			for xyz, copies in self._position_blocks(start, stop, step, times):
				block = slice(done, done + len(xyz))
				areas, pairs = self._atom_areas(xyz, copies)
				self._store(covered, areas, block)
				if pairs is not None:
					coincident.append(pairs)
				if free is not None: # the same positions without the environment
					self._store(free, self._atom_areas(xyz)[0], block)
				done += len(xyz)
				bar.update(len(xyz))
		if coincident: # once per run, not per block of frames
			universe = self.ag.universe
			warn_coincident(np.concatenate([pairs for _, pairs in coincident]),
			                label=lambda i: atom_label(universe.atoms[i]), frames=sum(n for n, _ in coincident))
		# results only replaced after a complete run
		self.results = self._to_results(covered, times, frames)
		self.free = self._to_results(free, times, frames) if free is not None else None
		self.buried = _difference(self.free, self.results) if free is not None else None
		elapsed = time.perf_counter() - tic
		logger.debug("SASA of %d frame(s) in %.2f s (%.1f frames/s)", len(frames), elapsed,
		             len(frames) / elapsed if elapsed > 0 else float('inf'))
		return self.results

	def _position_blocks(self, start: int | None, stop: int | None, step: int | None,
	                     times: list) -> Iterator[tuple[np.ndarray, list[Copies] | None]]:
		"""
		Positions (nm), (n_frames, n_active, 3), of the active atoms in blocks of frames of
		``start:stop:step``, with the environment atoms near the bodies in every frame (see
		:func:`environment_copies`; None without an environment); the time of every frame is
		appended to ``times``.
		"""
		trajectory = self.ag.universe.trajectory
		current = trajectory.ts.frame
		buffer, copies, n_copies = [], [], 0
		with_environment = self.environment is not None
		try:
			for ts in trajectory[start:stop:step]:
				xyz = to_nm(ts.positions[self._active_universe])
				buffer.append(xyz)
				times.append(ts.time)
				if with_environment: # only the environment atoms within reach: never the whole environment
					copies.append(environment_copies(xyz, self._active_body, to_nm(ts.positions[self.environment.indices]),
					                                 self._env_cutoff, box_nm(ts.dimensions)))
					n_copies += len(copies[-1].env)
				if (len(buffer) == self._frames_per_block or
				    n_copies and len(buffer) * len(self._active) + n_copies >= self._max_frame_atoms):
					yield np.array(buffer).reshape(len(buffer), len(self._active), 3), copies if with_environment else None
					buffer, copies, n_copies = [], [], 0
			if buffer:
				yield np.array(buffer).reshape(len(buffer), len(self._active), 3), copies if with_environment else None
		finally:
			trajectory[current] # leave the trajectory on the frame it was on

	def _atom_areas(self, xyz: np.ndarray, copies: list[Copies] | None = None
	                ) -> tuple[np.ndarray, tuple[int, np.ndarray] | None]:
		"""
		SASA (nm^2) of every active atom in a block of frames.

		Parameters
		----------
		xyz : np.ndarray
			(n_frames, n_active, 3) positions (nm) of the active atoms
		copies : list of Copies, optional
			of every frame the environment atoms near its bodies, which bury them; None:
			without the environment

		Returns
		-------
		areas : np.ndarray
			(n_frames, n_active)
		coincident : tuple or None
			(number of frames, (n_pairs, 2) universe indices) of the atoms at an identical
			position, None if there are none
		"""
		n_frames, n_atoms, _ = xyz.shape
		if not n_atoms:
			return np.zeros((n_frames, 0)), None

		# put every (frame, body) block in its own slab along x and seperate them,
		# so a _neighbor_list finds the neighbours of all of them and no pairs between them
		anchors = xyz[:, self._body_anchor] # (n_frames, n_bodies, 3) position of every body
		xyz = (xyz - xyz[:, self._active_anchor]).reshape(-1, 3)
		block = (np.arange(n_frames)[:, None] * self.n_bodies + self._active_body[None]).ravel()
		ext = np.tile(self._active_ext, n_frames)
		n_target = len(xyz)
		target = env = None
		if copies is not None: # every copy in the block (slab) of its body, which only it buries
			env = np.concatenate([c.env for c in copies])
			xyz = np.concatenate([xyz] + [c.positions - anchors[f, c.bodies] for f, c in enumerate(copies)])
			block = np.concatenate([block] + [f * self.n_bodies + c.bodies for f, c in enumerate(copies)])
			ext = np.concatenate([ext, self._env_ext[env]])
			target = np.arange(len(xyz)) < n_target
		span = 2 * np.abs(xyz).max() + 2 * ext.max() + 0.1 # nm
		xyz[:, 0] += block * span

		exposed, pairs = _exposed_counts(xyz, ext, block, self.dots, target)
		coincident = None
		if len(pairs):
			indices = self._active_universe[pairs % n_atoms] # universe indices; the copies: of their environment atom
			if env is not None:
				copy = pairs >= n_target
				indices[copy] = self.environment.indices[env[pairs[copy] - n_target]]
			coincident = (len(np.unique(pairs[:, 0] // n_atoms)), indices)
		return exposed[:n_target].reshape(n_frames, n_atoms) * self._dot_area, coincident

	def _new_arrays(self, n_frames: int) -> tuple[np.ndarray, np.ndarray | None]:
		"""Empty (body SASA (n_bodies, n_frames), atom SASA (n_atoms, n_frames) or None) of a run."""
		atoms = np.zeros((self.ag.n_atoms, n_frames)) if self.keep_atoms else None
		return np.zeros((self.n_bodies, n_frames)), atoms

	def _store(self, arrays: tuple[np.ndarray, np.ndarray | None], areas: np.ndarray, block: slice) -> None:
		"""Store the (n_frames, n_active) SASA of a block of frames: summed per body, and per atom if kept."""
		bodies, atoms = arrays
		n_frames = len(areas)
		index = (np.arange(n_frames)[:, None] * self.n_bodies + self._active_body[None]).ravel()
		bodies[:, block] = np.bincount(index, weights=areas.ravel(),
		                               minlength=n_frames * self.n_bodies).reshape(n_frames, -1).T
		if atoms is not None:
			atoms[self._active, block] = areas.T

	def _to_results(self, arrays: tuple[np.ndarray, np.ndarray | None], times: list, frames: np.ndarray) -> Results:
		bodies, atoms = arrays
		return Results(times, frames, bodies, self.body_labels, self.body_type_labels, atoms,
		               [atom_label(atom) for atom in self.ag] if atoms is not None else None,
		               atom_body=self._body_of_atom, ag=self.ag)

	# --- results
	def _require_results(self) -> Results:
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
		info = {'sassy': __version__, 'bodies': self.bodies, 'probe': self.probe, 'ndots': len(self.dots),
		        'cg': self.cg, 'n_atoms': self.ag.n_atoms,
		        'n_environment': None if self.environment is None else self.environment.n_atoms}
		save(fname, self._require_results(), self.free, meta=info | meta, table=table)

	# --- bodies
	def body_index(self, body: int | str) -> int:
		"""Index (0, 1, ...) of a body by its index (negative counts from the end) or label (e.g. 'GSH2')."""
		return body_index(self.body_labels, body)

	def body_atoms(self, body: int | str | None = None) -> mda.AtomGroup:
		"""Atoms of a body (index, negative counts from the end, or label); all atoms of ``ag`` for None."""
		if body is None:
			return self.ag
		return self.ag[self._members[self.body_index(body)]]

	# --- surfaces
	def surface(self, body: int | str = 0, frame: int | None = None,
	            reference: ArrayLike | mda.AtomGroup | None = None, ndots: int = SURFACE_DOTS,
	            body_type: str | None = None,
	            statistic: str | Callable[[np.ndarray], float] | float = 'median') -> Surface:
		"""
		The structure and its solvent accessible surface dots (Connolly-style dot surface) in
		one frame, with the radii and probe of this SASA; write it with :meth:`Surface.write`
		(PDB or GRO), see :func:`generate_surface`.

		The surface is that of one body (see ``bodies``): a molecule or residue on its own,
		or the whole AtomGroup with ``bodies='selection'``. With an environment, only the
		part left uncovered by it; the environment atoms within reach are kept too.

		Statistics are taken over the SASA distribution of the body's type only (all
		bodies of the type in all analysed frames; with ``bodies='selection'``: the total),
		so every type has its own median, mean, ... (see :meth:`Results.representative`).

		Parameters
		----------
		body : int or str
			index of the body (negative counts from the end) or its label (e.g. 'GSH2'),
			default the first; with ``bodies='selection'`` the whole AtomGroup is the only
			body (0, 'selection')
		frame : int, optional
			trajectory frame (negative counts from the end); default the analysed frame of the
			body whose SASA is closest to ``statistic`` (by default the median) of its
			type's distribution, i.e. a representative structure
		reference : np.ndarray or MDAnalysis.AtomGroup, optional
			reference structure, (n_atoms, 3) positions (nm) or an AtomGroup with as many
			atoms as the surface (in the same order, e.g. a mapped atomistic structure for
			a CG model): instead of ``frame``, the frame (of the last :meth:`run`, else of
			the whole trajectory) with the lowest RMSD to it is used, superposed onto it, so
			that both surfaces share one coordinate frame
		ndots : int
			minimum number of dots per atom (fewer than for the SASA, as dots are for viewing)
		body_type : str, optional
			instead of ``body``: a type (see :attr:`body_type_indices`); its representative
			body and frame (with ``frame``: the body closest to the statistic in that frame;
			with ``reference``: its first body)
		statistic : str, callable or float
			without ``frame`` and ``reference``: the representative frame is the one whose SASA
			is closest to this statistic of the type's SASA distribution: a name ('median',
			'mean', 'mode' (most probable SASA), 'q1', 'q3' (quartiles), 'min', 'max',
			'mean+std', 'mean-std', see :data:`STATISTICS`), a function of the SASA values
			(e.g. ``lambda sasa: np.percentile(sasa, 90)``), or a fixed SASA (nm^2)

		Returns
		-------
		Surface
			structure, dots, area (nm^2, from the dots), frame, ...

		Examples
		--------
		>>> sasa.surface(body_type='GSH').write('GSH_sasa.pdb')  # representative GSH and frame
		>>> sasa.surface('GSH2', frame=-1).write('last.gro')
		"""
		function, name = resolve_statistic(statistic) # checked also when not used
		if frame is not None: # negative counts from the end; IndexError if out of range
			frame = int(np.arange(self.ag.universe.trajectory.n_frames)[frame])

		# choose the body and frame
		chosen = None # remark on how they were chosen
		if body_type is not None and reference is not None: # the frame comes from the reference
			if body_type not in self.body_type_indices:
				raise KeyError(f"No body type {body_type!r}; types: {', '.join(self.body_type_indices)}.")
			body = int(self.body_type_indices[body_type][0])
		elif reference is None and (body_type is not None or frame is None): # from the statistic
			choice = {'body_type': body_type} if body_type is not None else {'body': body}
			body, frame, target = self._require_results().representative(statistic=function, frame=frame, **choice)
			chosen = chosen_remark(self.results, self.bodies, body, frame, target, name,
			                       body_choice=body_type is not None)
		else:
			body = self.body_index(body)

		atoms = self.body_atoms(body) # bodies='selection': all of ag
		in_ag = self._ag_position[atoms.indices]
		remarks = [body_remark(atoms, self.bodies, body, self.n_bodies,
		                       None if self.environment is None else self.environment.n_atoms)]
		remarks += ([chosen] if chosen else []) + radii_remarks(atoms, cg=self.cg, custom=self._custom_radii,
		                                                        probe=self.probe)
		environment = {} if self.environment is None else {'environment': self.environment,
		                                                   'environment_radii': self._env_radii}
		surface = generate_surface(atoms, radii=self.radii[in_ag], probe=self.probe, block=self._body_of_atom[in_ag],
		                           frame=frame, reference=reference, ndots=ndots, cg=self.cg, remarks=remarks,
		                           frames=None if self.results is None else self.results.frames, body=body,
		                           **environment)
		logger.info("surface of %s in frame %d: %d atoms, %d dots, SASA %.3f nm^2", self.body_labels[body],
		            surface.frame, surface.atoms.n_atoms, len(surface.dots), surface.area)
		return surface
