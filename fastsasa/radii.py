"""
Radii (nm) for the SASA of atomistic and Martini coarse-grained systems.

Atomistic van der Waals radii are those of Rowland & Taylor (1996),
doi:10.1021/jp953141+, except phosphorus, which is from Bondi (1964),
doi:10.1021/j100785a001. CG radii are the Martini 3 bead sizes (tiny, small,
regular); dummy beads have no radius. The probe radius is the radius of a
tiny bead.
"""
from collections.abc import Mapping
import logging
import numpy as np
import MDAnalysis as mda

logger = logging.getLogger(__name__)

AA_RADII = {
	'H': 0.109,
	'C': 0.175,
	'N': 0.161,
	'O': 0.156,
	'F': 0.144,
	'P': 0.180,
	'S': 0.179,
	'Cl': 0.174,
	'Br': 0.185,
	'I': 0.200,
}

CG_RADII = {
	'T': 0.191, # tiny bead
	'S': 0.230, # small bead
	'R': 0.264, # regular bead
	'U': 0.000, # dummy bead, no radius
}

PROBE_RADIUS = 0.191 # nm, tiny bead radius


def _symbol(sym: str) -> str:
	"""Element symbols in the usual capitalisation ('CL', 'cl' -> 'Cl'), so lookups ignore case."""
	return str(sym).strip().capitalize()

def _element_guesser():
	"""MDAnalysis element guess from an atom name (the guesser moved in MDAnalysis 2.8)."""
	try:
		from MDAnalysis.guesser.default_guesser import DefaultGuesser
		return DefaultGuesser(None).guess_atom_element
	except ImportError:
		from MDAnalysis.topology.guessers import guess_atom_element
		return guess_atom_element

def atom_symbols(ag: mda.AtomGroup, cg: bool = False) -> list[str]:
	'''
	Symbols used to look up the radius of each atom
	- AA: respective element. If the topology has no elements (e.g. gro, pdb without element column), 
		  they are guessed from the atom names with the MDAnalysis guesser ('CLA' -> 'Cl', 'SOD' -> 'Na'),
	 	  without changing ``ag``.
	- CG: bead size (R = regular, S = small, T = tiny, U = dummy) derived from atom type.
	'''
	if cg:
		return [at[0] if at[0] in ("S", "T", "U") else "R" for at in ag.types]
	if hasattr(ag.atoms, "elements"):
		return [_symbol(el if el else a.name[0]) for el, a in zip(ag.elements, ag)]
	logger.debug("no elements in the topology: guessing them from the atom names")
	guess = _element_guesser()
	guessed = {name: guess(name) for name in set(ag.names)}
	return [_symbol(guessed[name] or name[:1]) for name in ag.names]

def radii_remarks(ag: mda.AtomGroup, cg: bool = False, custom: bool = False,
                  probe: float = PROBE_RADIUS) -> list[str]:
	'''
	Short description of the radii used for ``ag`` and their sources, e.g. for the
	REMARK records of a written surface; ``custom`` notes radii given by the user.
	'''
	remarks = ["Custom radii"] if custom else []
	if cg:
		remarks.append(f"Martini CG radii: bead sizes tiny {CG_RADII['T']}, small {CG_RADII['S']}, "
		               f"regular {CG_RADII['R']} nm")
	else:
		remarks.append("Atomistic vdW radii: Rowland & Taylor (1996), doi:10.1021/jp953141+")
		if 'P' in atom_symbols(ag):
			remarks.append("Phosphorus radius: Bondi (1964), doi:10.1021/j100785a001")
	remarks.append(f"Probe radius {probe} nm")
	return remarks

def atom_radii(ag: mda.AtomGroup, cg: bool = False, table: Mapping[str, float] | None = None) -> np.ndarray:
	'''
	Van der Waals radii (nm) of the atoms in ``ag``, as used for the SASA.

	Parameters
	----------
	ag : MDAnalysis.AtomGroup
	cg : bool
		use the CG bead radii (keys T, S, R, U) instead of the atomistic ones (element symbols)
	table : mapping of str to float, optional
		radii (nm) by symbol (see :func:`atom_symbols`) that add to or replace the default
		radii (keys are case-insensitive), e.g. ``{'Na': 0.227, 'C': 0.170}``; a radius of 0 excludes an atom from the SASA calculation

	Returns
	-------
	np.ndarray
		(n_atoms,) radii of the atoms in ``ag``, in the same order
	'''
	# define radii table: default radii + user-specified radii
	defaults = CG_RADII if cg else AA_RADII
	custom = {} if table is None else {_symbol(sym): float(r) for sym, r in table.items()}

	# issue error if any radii are negative
	negative = [sym for sym, r in custom.items() if not r >= 0]
	if negative:
		raise ValueError(f"Radii must be >= 0 (nm), got {', '.join(f'{s}={custom[s]}' for s in negative)}.")
	
	table = {**{_symbol(sym): r for sym, r in defaults.items()}, **custom}
	
	symbols = atom_symbols(ag, cg=cg)

	# issue error if any symbols are missing from the radii table
	unknown = [i for i, sym in enumerate(symbols) if sym not in table]
	if unknown:
		kind = 'bead size' if cg else 'element'
		atoms = ', '.join(f"{ag[i].resname}{ag[i].resid}:{ag[i].name} ({kind} {symbols[i]!r})"
		                  for i in unknown[:10])
		more = f" and {len(unknown) - 10} more" if len(unknown) > 10 else ""
		missing = list(dict.fromkeys(symbols[i] for i in unknown))
		raise ValueError(f"No {'CG' if cg else 'atomistic'} SASA radius known for {len(unknown)} "
		                 f"{'bead' if cg else 'atom'}(s): {atoms}{more}. "
		                 f"Known {kind}s: {', '.join(table)}. Pass radii (nm) for the missing "
		                 f"{kind}s as a dict, e.g. radii={{{', '.join(f'{s!r}: ...' for s in missing)}}}.")
	if custom:
		logger.debug("custom radii (nm): %s", ', '.join(f'{s}={r}' for s, r in custom.items()))
	if logger.isEnabledFor(logging.DEBUG):
		used = dict(zip(*np.unique(symbols, return_counts=True)))
		logger.debug("radii (nm) used: %s", ', '.join(f'{s}={table[s]} ({n}x)' for s, n in used.items()))
	return np.array([table[sym] for sym in symbols])
