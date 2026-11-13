"""
Bodies of an AtomGroup: the sets of atoms that bury each other (molecules, residues or the
whole selection), their types and labels; and atom labels.
"""
import logging
from typing import Literal
import numpy as np
import MDAnalysis as mda

logger = logging.getLogger(__name__)

# Bodies
BODIES = ('molecule', 'residue', 'selection') # how an AtomGroup is split into bodies, see partition
Bodies = Literal['molecule', 'residue', 'selection']

def partition(ag: mda.AtomGroup, bodies: Bodies = 'molecule') -> tuple[np.ndarray, list[str], list[str]]:
	"""
	Split an AtomGroup into bodies: sets of atoms that bury each other.

	Parameters
	----------
	ag : MDAnalysis.AtomGroup
	bodies : str
		'molecule': every molecule; 'residue': every residue; 'selection': the whole AtomGroup as one body

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
		return index, types, numbered_labels(types)
	if bodies == 'residue':
		index = residue_index(ag)
		return index, residue_types(ag, index), residue_labels(ag, index)
	if bodies == 'selection':
		return np.zeros(ag.n_atoms, dtype=int), ['selection'] * bool(ag.n_atoms), ['selection'] * bool(ag.n_atoms)
	raise ValueError(f"bodies must be one of {', '.join(BODIES)}, got {bodies!r}.")


# Atoms
def atom_label(atom: mda.core.groups.Atom) -> str:
	'''Short name of an atom: ``<resname><resid>:<name>``, e.g. ``GSH1:SG``.'''
	resname = getattr(atom, 'resname', '')
	resid = getattr(atom, 'resid', atom.resindex + 1)
	return f"{resname}{resid}:{getattr(atom, 'name', atom.index)}"


# Molecules
def molecule_index(ag: mda.AtomGroup) -> np.ndarray:
	'''
	Molecule index of every atom in ag, 0, 1, ... in the order in which the molecules appear.

	The molecules are taken, in this order of preference, from

	1. the molnums of the topology (e.g. tpr),
	2. the bonds of the topology (e.g. psf, PDB with CONECT records): every set of bonded
	   atoms (fragment) is a molecule, so any mix of molecules is split,
	3. without atom names: the residues,
	4. else ``ag`` is taken as consecutive copies of one molecule: the shortest repeating
	   unit of atoms with the same atom names, residue names and resids (up to an offset
	   per copy) in the same order. If nothing repeats, a single residue is a single
	   molecule; several residues raise a ValueError (use a tpr, or change the selection).
	'''
	if ag.n_atoms == 0:
		return np.zeros(0, dtype=int)
	if hasattr(ag.atoms, 'molnums'):
		logger.debug("molecules from the molnums of the topology")
		return _rank(ag.molnums)
	if _has_bonds(ag):
		logger.debug("molecules from the bonds of the topology (fragments)")
		return _rank(ag.fragindices)
	if not hasattr(ag.atoms, 'names'):
		logger.debug("molecules from the residues of the topology")
		return _rank(ag.resindices)
	return np.arange(ag.n_atoms) // max(1, _copy_length(ag))

def molecule_types(ag: mda.AtomGroup, molecules: np.ndarray | None = None) -> list[str]:
	'''
	Type name of every molecule: its moltype (tpr), else
	the residue name of its first atom, for molecules with the same atom names and residue
	names in the same order. Different molecules starting with the same residue name are
	told apart by a suffix (e.g. ALA, ALA_2).

	Parameters
	----------
	ag : MDAnalysis.AtomGroup
	molecules : np.ndarray, optional
		molecule index of every atom; default :func:`molecule_index`

	Returns
	-------
	list of str
		(n_molecules,) type names
	'''
	molecules = molecule_index(ag) if molecules is None else np.asarray(molecules)
	if hasattr(ag.atoms, 'moltypes'):
		first = np.unique(molecules, return_index=True)[1]
		return [str(name) for name in ag.moltypes[first]]
	return _signature_types(ag, molecules, 'molecule')

def _has_bonds(ag: mda.AtomGroup) -> bool:
	'''Whether the topology has bonds (e.g. tpr, psf, PDB with CONECT records).'''
	try:
		return hasattr(ag.universe.atoms, 'bonds') and len(ag.universe.bonds) > 0
	except Exception: # e.g. no bond information after all
		return False

def _copy_length(ag: mda.AtomGroup) -> int:
	'''
	Number of atoms in the repeating molecule: the shortest unit of which ``ag`` consists
	of consecutive copies, every copy with the same atom names and residue names in the
	same order and the same resids up to an offset. Without a repeating unit, a single
	residue is one molecule (all its atoms); several residues raise a ValueError.
	'''
	n = ag.n_atoms
	if n == 0:
		return 0

	# convert names and resnames to unique integers
	labels = [np.unique(getattr(ag, attr), return_inverse=True)[1].ravel()
	          for attr in ('names', 'resnames') if hasattr(ag.atoms, attr)]
	resids = np.asarray(ag.resids if hasattr(ag.atoms, 'resids') else ag.resindices, dtype=np.int64)

	# find shortest length that repeats to make the whole AtomGroup
	for length in range(1, n):
		if n % length:
			continue
		copies = n // length
		# check if every copy has the same names/resnames and resids
		if not all((label.reshape(copies, length) == label[:length]).all() for label in labels):
			continue
		relative = resids.reshape(copies, length) - resids[::length, None]
		if (relative == relative[0]).all():
			logger.debug("molecules from repeating units: %d copies of %d atoms", copies, length)
			return length
	if len(np.unique(ag.resindices)) == 1: # a single residue: a single molecule
		logger.debug("molecules: no repeating unit in %d atoms of one residue, taken as one molecule", n)
		return n
	raise ValueError(f"No molecules found: the topology has neither molecules (molnums, e.g. tpr) nor bonds, and "
	                 f"the {n} atoms of the selection are no consecutive copies of one molecule. Use a tpr (or a "
	                 f"topology with bonds), or change the selection (e.g. to one molecule type); or choose the "
	                 f"bodies explicitly: bodies='selection' or 'residue' (-bodies selection / residue).")


# Residues
def residue_index(ag: mda.AtomGroup) -> np.ndarray:
	'''Residue index of every atom in ag, 0, 1, ... in the order in which the residues appear.'''
	return _rank(ag.resindices) if ag.n_atoms else np.zeros(0, dtype=int)

def residue_types(ag: mda.AtomGroup, residues: np.ndarray | None = None) -> list[str]:
	'''
	Type name of every residue (see :func:`residue_index`): its residue name, for residues
	with the same atom names in the same order; residues of the same name with other atoms
	(e.g. termini) are told apart by a suffix (ALA, ALA_2), so that all residues of a type
	have the same atoms.
	'''
	residues = residue_index(ag) if residues is None else np.asarray(residues)
	return _signature_types(ag, residues, 'residue')

def residue_labels(ag: mda.AtomGroup, residues: np.ndarray | None = None) -> list[str]:
	'''
	Label of every residue (see :func:`residue_index`): ``<resname><resid>`` as in
	:func:`atom_label` (e.g. ALA12); residues with the same label in different segments
	(e.g. chains) get their segid appended (ALA12:A), and if that does not tell them apart,
	a number (ALA12_2).
	'''
	residues = residue_index(ag) if residues is None else np.asarray(residues)
	if not len(residues):
		return []
	first = ag[np.unique(residues, return_index=True)[1]] # first atom of every residue, in order
	resnames = first.resnames if hasattr(ag.atoms, 'resnames') else [''] * len(first)
	resids = first.resids if hasattr(ag.atoms, 'resids') else first.resindices + 1
	labels = [f"{name}{resid}" for name, resid in zip(resnames, resids)]
	if len(set(labels)) < len(labels) and hasattr(ag.atoms, 'segids'):
		segids = {} # label -> the segids of the residues with that label
		for label, segid in zip(labels, first.segids):
			segids.setdefault(label, set()).add(segid)
		# the segid only where it tells residues of the same label apart
		labels = [f"{label}:{segid}" if len(segids[label]) > 1 and segid else label
		          for label, segid in zip(labels, first.segids)]
	seen, unique = {}, []
	for label in labels:
		seen[label] = seen.get(label, 0) + 1
		unique.append(label if seen[label] == 1 else f"{label}_{seen[label]}")
	return unique


# Helpers for any index: molecules, residues or bodies
def members(index: np.ndarray, n: int | None = None) -> list[np.ndarray]:
	'''
	The atoms of every body: inverts a per-atom index (which body every atom belongs to)
	into a list of the atoms of every body.

	Parameters
	----------
	index : np.ndarray
		(n_atoms,) index of body for every atom
	n : int, optional
		number of bodies

	Returns
	-------
	list of np.ndarray
		(n,) for every body the index of its atoms, in increasing order; ``ag[members(index)[k]]`` are the atoms of body k
	'''
	index = np.asarray(index, dtype=np.int64)
	if n is None:
		n = int(index.max()) + 1 if len(index) else 0
	order = np.argsort(index, kind='stable')
	return np.split(order, np.cumsum(np.bincount(index, minlength=n))[:-1]) if n else []

def type_members(types: list[str]) -> dict[str, np.ndarray]:
	'''
	Indices of every type: {type: indices of its entries in ``types``}, types in the order of
	first appearance, e.g. ['GSH', 'SOL', 'GSH'] -> {'GSH': array([0, 2]), 'SOL': array([1])}.
	'''
	indices = {}
	for i, name in enumerate(types):
		indices.setdefault(name, []).append(i)
	return {name: np.array(found) for name, found in indices.items()}

def numbered_labels(types: list[str]) -> list[str]:
	'''
	Label of every set from its type: the type numbered from 1 within the type, e.g.
	['GSH', 'GSH', 'SOL'] -> ['GSH1', 'GSH2', 'SOL1'].
	'''
	counts, labels = {}, []
	for name in types:
		counts[name] = counts.get(name, 0) + 1
		labels.append(f"{name}{counts[name]}")
	return labels

def _rank(keys: np.ndarray) -> np.ndarray:
	'''0, 1, ... for the distinct keys in the order of their first appearance.'''
	_, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
	rank = np.empty(len(first), dtype=int)
	rank[np.argsort(first)] = np.arange(len(first))
	return rank[inverse.ravel()]

def _signature_types(ag: mda.AtomGroup, index: np.ndarray, fallback: str) -> list[str]:
	"""
	Type name of every set of atoms of ``index``: the residue name of its first
	atom (``fallback`` without residue names), for sets with the same atom names and residue
	names in the same order; different sets with the same first residue name are told
	apart by a suffix (e.g. ALA, ALA_2).
	"""
	n = int(index.max()) + 1 if len(index) else 0
	values = [getattr(ag, attr) for attr in ('names', 'resnames') if hasattr(ag.atoms, attr)]
	resnames = ag.resnames if hasattr(ag.atoms, 'resnames') else None
	names, signatures, bases = [], {}, {} # signature (names, resnames) -> type name; base -> count
	for atoms in members(index, n):
		signature = tuple(tuple(value[atoms].tolist()) for value in values)
		if signature not in signatures:
			base = str(resnames[atoms[0]]) if resnames is not None else fallback
			bases[base] = bases.get(base, 0) + 1
			signatures[signature] = base if bases[base] == 1 else f"{base}_{bases[base]}"
		names.append(signatures[signature])
	return names
