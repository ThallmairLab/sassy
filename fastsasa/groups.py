"""
Molecules of an AtomGroup: which atoms bury each other.
"""
import logging
import numpy as np
import MDAnalysis as mda

logger = logging.getLogger(__name__)


def molecule_index(ag: mda.AtomGroup) -> np.ndarray:
	'''
	Molecule index of ever atom in ag in the order in which they appear.

	Molecules are taken from the molnums of the topology (e.g. tpr). Without
	molnums, ``ag`` is taken as consecutive copies of one molecule: the shortest
	repeating unit of atoms with the same atom names, residue names and resids
	(up to an offset per copy) in the same order, see :func:`copy_length`. If
	nothing repeats, ``ag`` is a single molecule. Without atom
	names, every residue is a molecule.
	'''
	if hasattr(ag.atoms, 'molnums') or not hasattr(ag.atoms, 'names'):
		logger.debug("molecules from the %s of the topology", 'molnums' if hasattr(ag.atoms, 'molnums') else 'residues')
		keys = ag.molnums if hasattr(ag.atoms, 'molnums') else ag.resindices
		_, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
		rank = np.empty(len(first), dtype=int)
		rank[np.argsort(first)] = np.arange(len(first))
		return rank[inverse.ravel()]
	return np.arange(ag.n_atoms) // max(1, copy_length(ag))

def copy_length(ag: mda.AtomGroup) -> int:
	'''
	Number of atoms in the repeating molecule.
	Molecule is  the shortest unit of which ``ag`` consists of consecutive
	copies: every copy has the same atom names and residue names in the same
	order, and the same resids order of resids.
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
	logger.warning("could not find a repeating unit in %d atoms; treating the whole AtomGroup as one molecule "
	               "(give a topology with molecules, e.g. a tpr, or select one molecule type)", n)
	return n

def split_molecules(ag: mda.AtomGroup) -> list[mda.AtomGroup]:
	'''
	Split ``ag`` into its molecules (see :func:`molecule_index`), in the order in
	which they appear.
	'''
	molecules = molecule_index(ag)
	return [ag[molecules == i] for i in range(molecules.max() + 1 if len(molecules) else 0)]

def n_molecule_types(ag: mda.AtomGroup) -> int:
	'''
	Number of different molecule types in ``ag``: by the moltypes of the topology
	(tpr), else by the sequence of atom names and residue names of every
	molecule (see :func:`molecule_index`).
	'''
	if ag.n_atoms == 0:
		return 0
	if hasattr(ag.atoms, 'moltypes'):
		return len(np.unique(ag.moltypes))
	molecules = molecule_index(ag)
	attrs = [attr for attr in ('names', 'resnames') if hasattr(ag.atoms, attr)]
	types = {tuple(tuple(getattr(ag[molecules == i], attr)) for attr in attrs)
	         for i in range(molecules.max() + 1)}
	return len(types)
