"""
Results of a SASA calculation: the SASA (nm^2) of every body, of every atom and in
total over the analysed frames, with their labels, and saving to / loading from npz.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import MDAnalysis as mda

from .groups import body_type_indices

COLUMNS = ('bodies', 'total', 'types', 'atoms') # the levels of Results.table


def body_index(labels: list[str], body: int | str) -> int:
	"""Index (0, 1, ...) of a body by its index (negative counts from the end) or its label in ``labels``."""
	n = len(labels)
	if isinstance(body, str):
		if body not in labels:
			raise KeyError(f"No body {body!r}; labels: {', '.join(labels[:10])}{', ...' if n > 10 else ''}.")
		return labels.index(body)
	if not -n <= body < n:
		whole = " (the whole selection is one body, index 0)" if labels == ['selection'] else ""
		raise IndexError(f"Body index {body} out of range for {n} body(ies){whole}.")
	return int(body) % n


@dataclass(repr=False)
class Results:
	"""
	SASA (nm^2) of one quantity of one :meth:`SASA.run`: of every body (molecule, residue
	or the whole selection, see ``bodies`` of :class:`SASA`), optionally of every atom,
	over the analysed frames.

	:class:`SASA` gives three of them after a run: :attr:`SASA.results` (the SASA, with an
	environment the part it leaves uncovered), and with ``buried=True`` :attr:`SASA.free`
	(without the environment) and :attr:`SASA.buried` (the area the environment buries,
	free minus results). All have the same attributes and methods.

	Attributes
	----------
	times, frames : np.ndarray
		(n_frames,) times (ps) and trajectory frames
	body_sasa : np.ndarray
		(n_bodies, n_frames) SASA of every body
	body_labels : list of str
		(n_bodies,) unique label of every body (e.g. GSH2, ALA12, 'selection')
	body_type_labels : list of str
		(n_bodies,) type of every body (e.g. GSH, ALA, 'selection')
	atom_sasa : np.ndarray or None
		(n_atoms, n_frames) SASA of every atom of the AtomGroup, in its order; atoms with
		radius 0 have SASA 0; None if not kept (``keep_atoms=False``)
	atom_labels : list of str or None
		(n_atoms,) label of every atom, ``<resname><resid>:<name>``
	ag : MDAnalysis.AtomGroup or None
		the atoms, for :meth:`select` by selection string or AtomGroup; None after
		:func:`load` without a universe

	Examples
	--------
	>>> res = SASA(u.select_atoms('moltype GSH')).run()
	>>> res.total_sasa                 # (n_frames,)
	>>> res.body_sasa, res.body_labels # (n_bodies, n_frames), ['GSH1', 'GSH2', ...]
	>>> res.body('GSH2')               # (n_frames,)
	>>> res.type_sasa()                # {'GSH': (n_frames,), ...}
	>>> res.select('name S*')     # (n_frames,) summed over the selected atoms
	>>> res.table(['bodies', 'total'])  # labels and (n_columns, n_frames) values
	"""
	times: np.ndarray
	frames: np.ndarray
	body_sasa: np.ndarray
	body_labels: list[str]
	body_type_labels: list[str]
	atom_sasa: np.ndarray | None = None
	atom_labels: list[str] | None = None
	ag: mda.AtomGroup | None = field(default=None, compare=False)

	def __post_init__(self) -> None:
		self.times = np.asarray(self.times, dtype=float)
		self.frames = np.asarray(self.frames, dtype=int)
		self.body_sasa = np.asarray(self.body_sasa, dtype=float).reshape(len(self.body_labels), len(self.frames))
		self.body_labels, self.body_type_labels = list(self.body_labels), list(self.body_type_labels)
		if self.atom_sasa is not None:
			self.atom_sasa = np.asarray(self.atom_sasa, dtype=float).reshape(-1, len(self.frames))
		if self.atom_labels is not None:
			self.atom_labels = list(self.atom_labels)

	def __repr__(self) -> str:
		types = ', '.join(f"{name} x{len(indices)}" for name, indices in self.body_type_indices.items())
		atoms = f"{len(self.atom_sasa)} atoms" if self.atom_sasa is not None else "no atoms kept"
		bodies = 'body' if self.n_bodies == 1 else 'bodies'
		return f"<Results: {self.n_bodies} {bodies} ({types}), {self.n_frames} frames, {atoms}>"

	@property
	def n_bodies(self) -> int:
		return len(self.body_labels)

	@property
	def n_frames(self) -> int:
		return len(self.frames)

	@property
	def total_sasa(self) -> np.ndarray:
		"""(n_frames,) SASA of all bodies together."""
		return self.body_sasa.sum(axis=0)

	@property
	def body_type_indices(self) -> dict[str, np.ndarray]:
		"""Body indices of every type, in the order of first appearance: {type: indices}."""
		return body_type_indices(self.body_type_labels)

	# bodies
	def body_index(self, body: int | str) -> int:
		"""Index (0, 1, ...) of a body by its index (negative counts from the end) or label (e.g. 'GSH2')."""
		return body_index(self.body_labels, body)

	def body(self, body: int | str | ArrayLike) -> np.ndarray:
		"""
		SASA of bodies by index (negative counts from the end), label (e.g. 'GSH2') or
		indices: (n_frames,) for one body, else (n_indices, n_frames).
		"""
		if isinstance(body, (str, int, np.integer)):
			return self.body_sasa[self.body_index(body)]
		index = np.asarray(body)
		if not np.issubdtype(index.dtype, np.integer):
			raise TypeError(f"Body indices must be integers, got {index.dtype}.")
		if ((index < -self.n_bodies) | (index >= self.n_bodies)).any():
			raise IndexError(f"Body index out of range for {self.n_bodies} body(ies): {index.tolist()}.")
		return self.body_sasa[index]

	def type_sasa(self, per_body: bool = False) -> dict[str, np.ndarray]:
		"""
		SASA of every body type (see :attr:`body_type_indices`): summed over its bodies (n_frames,), or
		with ``per_body`` the SASA of each of them (n_bodies_of_type, n_frames).
		"""
		return {name: self.body_sasa[indices] if per_body else self.body_sasa[indices].sum(axis=0)
		        for name, indices in self.body_type_indices.items()}

	# atoms
	def _atom_indices(self, selection: str | mda.AtomGroup | ArrayLike) -> np.ndarray:
		"""Indices in the atoms of a selection string (applied to ``ag``), an AtomGroup or indices."""
		if isinstance(selection, str) or hasattr(selection, 'atoms') or isinstance(selection, mda.core.groups.Atom):
			if self.ag is None:
				raise ValueError("Selecting by a selection string or AtomGroup needs the atoms: give the "
				                 "universe to load(), or select by atom indices.")
			if isinstance(selection, str):
				selection = self.ag.select_atoms(selection)
			elif isinstance(selection, mda.core.groups.Atom):
				selection = selection.universe.atoms[[selection.index]]
			else: # AtomGroup, Residue(Group), Segment(Group)
				selection = selection.atoms
			if selection.universe is not self.ag.universe:
				raise ValueError("The selection belongs to a different Universe than the atoms of the SASA.")
			lookup = np.full(self.ag.universe.atoms.n_atoms, -1, dtype=np.int64)
			lookup[self.ag.indices] = np.arange(self.ag.n_atoms)
			indices = lookup[selection.indices]
			if (indices < 0).any():
				missing = selection.indices[indices < 0]
				raise ValueError(f"{len(missing)} atom(s) of the selection are not in the AtomGroup of the SASA: "
				                 f"universe indices {missing[:10].tolist()}{' ...' if len(missing) > 10 else ''}.")
			return indices
		indices = np.asarray(selection)
		if not np.issubdtype(indices.dtype, np.integer):
			raise TypeError(f"selection must be a selection string, an AtomGroup or atom indices, got "
			                f"{type(selection).__name__}.")
		return indices

	def select(self, selection: str | mda.AtomGroup | ArrayLike, per_atom: bool = False) -> np.ndarray:
		"""
		SASA of a selection of the atoms: summed (n_frames,), or with ``per_atom`` of every
		selected atom (n_selected, n_frames), in the order of the selection.

		The SASA of every atom is that within its own body (and environment), so the SASA of
		a selection is the part of the bodies' surface that belongs to its atoms, not the
		SASA of the selected atoms on their own. Needs the per-atom SASA (``keep_atoms``).

		Parameters
		----------
		selection : str, MDAnalysis.AtomGroup or array-like of int
			selection string applied to the atoms (e.g. ``'name S*'``), an AtomGroup whose
			atoms are all among them, or indices in the atoms
		"""
		if self.atom_sasa is None:
			raise RuntimeError("The SASA of every atom was not kept (keep_atoms=False): only the bodies "
			                   "and the total are available.")
		values = self.atom_sasa[self._atom_indices(selection)]
		return values if per_atom else values.sum(axis=0)

	# tables
	def table(self, columns: Sequence[str] = ('bodies',), groups: Sequence[str | mda.AtomGroup] = (),
	          suffix: str = '') -> tuple[list[str], np.ndarray]:
		"""
		Columns of SASA over time, e.g. for a text file or plots.

		Parameters
		----------
		columns : sequence of str
			levels, in this order: 'bodies' (one column per body), 'total', 'types' (one
			column per body type, summed over its bodies), 'atoms' (one column per atom)
		groups : sequence of str or AtomGroup
			after them, one column per group: its summed SASA, see :meth:`select`
		suffix : str
			appended to every label (e.g. ' buried')

		Returns
		-------
		labels : list of str
		values : np.ndarray
			(n_columns, n_frames)
		"""
		labels, values = [], []
		for column in columns:
			if column == 'bodies':
				labels += self.body_labels
				values += list(self.body_sasa)
			elif column == 'total':
				labels.append('total')
				values.append(self.total_sasa)
			elif column == 'types':
				for name, value in self.type_sasa().items():
					labels.append(name)
					values.append(value)
			elif column == 'atoms':
				if self.atom_sasa is None:
					raise RuntimeError("The SASA of every atom was not kept (keep_atoms=False).")
				labels += self.atom_labels
				values += list(self.atom_sasa)
			else:
				raise ValueError(f"Unknown column {column!r}; columns: {', '.join(COLUMNS)}.")
		for group in groups:
			labels.append(group if isinstance(group, str) else f"{len(group)} atoms")
			values.append(self.select(group))
		return [f"{label}{suffix}" for label in labels], np.array(values).reshape(len(values), self.n_frames)

	def save(self, fname: str | Path, **meta) -> None:
		"""Save to an npz file (see :func:`save`), readable with :func:`load`."""
		save(fname, self, meta=meta)


def _difference(minuend: Results, subtrahend: Results) -> Results:
	"""Element-wise ``minuend - subtrahend`` (e.g. the buried area: free minus covered)."""
	atoms = None
	if minuend.atom_sasa is not None and subtrahend.atom_sasa is not None:
		atoms = minuend.atom_sasa - subtrahend.atom_sasa
	return Results(minuend.times, minuend.frames, minuend.body_sasa - subtrahend.body_sasa, minuend.body_labels,
	               minuend.body_type_labels, atoms, minuend.atom_labels, minuend.ag)


@dataclass(repr=False)
class Saved:
	"""
	Results loaded with :func:`load`: the same as after :meth:`SASA.run`.

	Attributes
	----------
	results : Results
		the SASA (with an environment: the part it leaves uncovered)
	free, buried : Results or None
		without the environment, and the area it buries (free minus results); None if not saved
	meta : dict
		how the SASA was computed (sassy version, bodies, probe, ndots, ...)
	"""
	results: Results
	free: Results | None = None
	buried: Results | None = None
	meta: dict = field(default_factory=dict)

	def __repr__(self) -> str:
		return f"<Saved: {self.results!r}{', with free and buried' if self.free is not None else ''}>"


def save(fname: str | Path, results: Results, free: Results | None = None, meta: dict | None = None,
         table: tuple[list[str], np.ndarray] | None = None) -> None:
	"""
	Save results to an npz file, readable with :func:`load` (and with ``numpy.load``).

	Keys: ``times``, ``frames``, ``body_labels``, ``body_type_labels``, ``body_sasa`` (n_bodies,
	n_frames), ``total_sasa``; with the per-atom SASA ``atom_sasa``, ``atom_labels`` and ``atom_indices`` (universe
	indices); with ``free`` the same arrays prefixed ``free_``; ``meta`` (JSON); with a
	``table`` (labels, values) ``columns`` and ``column_labels``.
	"""
	arrays = {'times': results.times, 'frames': results.frames, 'body_labels': np.array(results.body_labels, dtype=str),
	          'body_type_labels': np.array(results.body_type_labels, dtype=str)}
	for prefix, item in (('', results), ('free_', free)):
		if item is None:
			continue
		arrays[f'{prefix}body_sasa'] = item.body_sasa
		arrays[f'{prefix}total_sasa'] = item.total_sasa
		if item.atom_sasa is not None:
			arrays[f'{prefix}atom_sasa'] = item.atom_sasa
	if results.atom_labels is not None:
		arrays['atom_labels'] = np.array(results.atom_labels, dtype=str)
	if results.ag is not None:
		arrays['atom_indices'] = results.ag.indices
	if table is not None:
		arrays['column_labels'] = np.array(table[0], dtype=str)
		arrays['columns'] = table[1]
	arrays['meta'] = np.array(json.dumps(meta or {}))
	with open(fname, 'wb') as out: # a file object: np.savez would append .npz to e.g. x.NPZ
		np.savez(out, **arrays)


def load(fname: str | Path, universe: mda.Universe | None = None) -> Saved:
	"""
	Load results saved with :meth:`SASA.save` (or ``sassy -o x.npz``).

	Parameters
	----------
	fname : str or Path
		npz file
	universe : MDAnalysis.Universe, optional
		the universe of the SASA, to select atoms by selection string or AtomGroup (see
		:meth:`Results.select`); without it, atoms are selected by index

	Returns
	-------
	Saved
		``.results``, ``.free``, ``.buried`` (as after :meth:`SASA.run`) and ``.meta``
	"""
	with np.load(fname, allow_pickle=False) as data:
		data = dict(data)
	ag = None
	if universe is not None and 'atom_indices' in data:
		indices = data['atom_indices']
		if len(indices) and indices.max() >= universe.atoms.n_atoms:
			raise ValueError(f"The universe has {universe.atoms.n_atoms} atoms, too few for the saved atoms.")
		ag = universe.atoms[indices]
	atom_labels = data['atom_labels'].tolist() if 'atom_labels' in data else None

	def results(prefix: str) -> Results:
		return Results(data['times'], data['frames'], data[f'{prefix}body_sasa'], data['body_labels'].tolist(),
		               data['body_type_labels'].tolist(), data.get(f'{prefix}atom_sasa'), atom_labels, ag)

	main = results('')
	free = results('free_') if 'free_body_sasa' in data else None
	return Saved(main, free, _difference(free, main) if free is not None else None,
	             json.loads(str(data['meta'])) if 'meta' in data else {})
