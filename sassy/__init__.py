"""
sassy: solvent accessible surface areas (SASA)
for every molecule, or every atom, of an MDAnalysis AtomGroup over a trajectory.

Every atom is a sphere of its van der Waals radius plus the probe radius,
covered with geodesic dots; a dot is exposed if it is not
inside the sphere of another atom of the same body (Shrake-Rupley).
By default every molecule is a body on its own, so the SASA describes the shape of
every molecule, not its exposure in the surrounding system; residues or the whole
selection can be bodies instead, and an environment (e.g. a protein around a ligand)
can cover every body without being counted.

As in MDAnalysis, "atom" means every particle of the AtomGroup: an atom in
atomistic systems, a bead in coarse-grained ones.

>>> import MDAnalysis as mda
>>> from sassy import SASA
>>> u = mda.Universe('topol.tpr', 'traj.xtc')
>>> sasa = SASA(u.select_atoms('moltype GSH')).run()   # (n_molecules, n_frames), nm^2
>>> ligand = SASA(u.select_atoms('resname LIG'), environment='protein', buried=True)
"""
from .dots import unit_dots, ico_dot_arc, ico_dot_dod
from .engine import surface_dots, warn_coincident
from .environment import environment_copies
from .groups import (copy_length, molecule_index, molecule_labels, molecule_types, n_molecule_types, split_molecules,
                     residue_index, residue_labels, residue_types)
from .radii import AA_RADII, CG_RADII, PROBE_RADIUS, atom_radii, atom_symbols, radii_remarks
from .sasa import BODIES, N_DOTS, SASA, partition
from .units import to_angstrom, to_nm
from .surface import SURFACE_DOTS, Surface, STATISTICS, best_matching_frame, closest_to, generate_surface

import importlib.metadata
try: # the version of pyproject.toml, from the installed package metadata
	__version__ = importlib.metadata.version("sassy")
except importlib.metadata.PackageNotFoundError: # run from the source tree without installing
	__version__ = "0+unknown"

__all__ = [
	"SASA", "N_DOTS", "BODIES", "partition", "surface_dots", "warn_coincident", "environment_copies",
	"to_nm", "to_angstrom",
	"Surface", "SURFACE_DOTS", "STATISTICS", "best_matching_frame", "closest_to", "generate_surface",
	"unit_dots", "ico_dot_arc", "ico_dot_dod",
	"copy_length", "molecule_index", "molecule_labels", "molecule_types", "n_molecule_types", "split_molecules",
	"residue_index", "residue_labels", "residue_types",
	"AA_RADII", "CG_RADII", "PROBE_RADIUS", "atom_radii", "atom_symbols", "radii_remarks",
]
