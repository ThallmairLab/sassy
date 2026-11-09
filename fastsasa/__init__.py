"""
fastsasa: solvent accessible surface areas (SASA)
for every molecule, or every atom, of an MDAnalysis AtomGroup over a trajectory.

Every atom is a sphere of its van der Waals radius plus the probe radius,
covered with geodesic dots; a dot is exposed if it is not
inside the sphere of another atom of the same molecule (Shrake-Rupley).
Molecules are treated on their own, so the SASA describes the shape of every
molecule, not its exposure in the surrounding system.

As in MDAnalysis, "atom" means every particle of the AtomGroup: an atom in
atomistic systems, a bead in coarse-grained ones.

>>> import MDAnalysis as mda
>>> from fastsasa import SASA
>>> u = mda.Universe('topol.tpr', 'traj.xtc')
>>> sasa = SASA(u.select_atoms('moltype GSH')).run()   # (n_molecules, n_frames), nm^2
"""
from .dots import unit_dots, ico_dot_arc, ico_dot_dod
from .engine import surface_dots
from .groups import copy_length, molecule_index, molecule_labels, molecule_types, n_molecule_types, split_molecules
from .radii import AA_RADII, CG_RADII, PROBE_RADIUS, atom_radii, atom_symbols, radii_remarks
from .sasa import N_DOTS, SASA
from .units import to_angstrom, to_nm
from .surface import SURFACE_DOTS, Surface, STATISTICS, best_matching_frame, closest_to, generate_surface

__version__ = "0.1.0"

__all__ = [
	"SASA", "N_DOTS", "surface_dots", "to_nm", "to_angstrom",
	"Surface", "SURFACE_DOTS", "STATISTICS", "best_matching_frame", "closest_to", "generate_surface",
	"unit_dots", "ico_dot_arc", "ico_dot_dod",
	"copy_length", "molecule_index", "molecule_labels", "molecule_types", "n_molecule_types", "split_molecules",
	"AA_RADII", "CG_RADII", "PROBE_RADIUS", "atom_radii", "atom_symbols", "radii_remarks",
]
