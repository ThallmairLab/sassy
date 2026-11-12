# sassy

Calculates the solvent accessible surface areas (SASA) of molecules and atoms over trajectories. Supports atomistic and coarse-grained (CG) Martini 3 trajectories.
Uses a fast Shrake-Rupley algorithm on a [MDAnalysis](https://www.mdanalysis.org) AtomGroup.

## Installation

```bash
git clone ...
pip install .     
```

Requires Python ≥ 3.11, numpy, scipy, numba, MDAnalysis and matplotlib.

## Usage
### Python API
```python
import MDAnalysis as mda
from sassy import SASA

u = mda.Universe("topol.tpr", "traj.xtc")

sasa = SASA(u.select_atoms("moltype GSH"))  # initialize SASA object
sasa.run(verbose=True)						# perform calculation with progress bar
sasa.get_sasa("name S*")               		# (n_frames,) summed over the selection
```
### Command line

```bash
# one column per molecule, each on its own, as xvg, saves plots in plots directory
sassy -s topol.tpr -f traj.xtc -select "moltype GSH" -total -o sasa.xvg -plots plots

# SASA of the whole selection (its molecules bury each other)
sassy -s topol.tpr -f traj.xtc -select "resname GSH" -whole -o sasa.xvg -plots plots
```

### Surface dots

The structure of one frame and its solvent accessible surface dots (a Connolly-style
dot surface) can be written as a PDB file for viewing: atoms as
ATOM records, dots as HETATM records named `DOT` with the residue
name and id of the atom they belong to (select them e.g. with `name DOT`).

The command line writes one surface per molecule type, of a representative
molecule and frame of that type: statistics are taken over the SASA distribution of the
type only (all its molecules in all analysed frames). The statistic is
the median by default, or can be specified using `-surface-stat`: `mean`, `mode` (the most probable
SASA), the quartiles `q1`/`q3`, `min`, `max`, `mean+std`, `mean-std`, a custom function or a fixed SASA
value (in nm²). Instead, a given frame (`-surface-frame`: per type the molecule closest to the
median in that frame) or the frame with the lowest RMSD to a reference structure (same
atoms; one type or `-surface-molecule`), superposed onto it.

```python
sasa = SASA(u.select_atoms("moltype GSH"))
sasa.run()
sasa.write_surface("GSH_sasa.pdb", molecule_type="GSH")  # representative GSH and frame
sasa.write_surface("dots.pdb", statistic="mean+std")           # frame closest to mean + std
```

```bash
sassy -s topol.tpr -f traj.xtc -select "moltype GSH or moltype SOL" -surface surfaces  					  # surfaces/GSH_sasa.pdb, surfaces/SOL_sasa.pdb
sassy -s topol.tpr -f traj.xtc -select "moltype GSH" -surface -surface-molecule GSH1 -surface-stat mean   # ./GSH1_sasa.pdb
sassy -s cg.tpr -f cg.xtc -select "moltype GSH" -cg -surface cg -surface-ref mapped.pdb					  # clodest to reference structure
```

Surfaces use fewer dots (`ndots` / `-surface-ndots`, default 240) than the SASA.

## Method

As in MDAnalysis, "atom" means every particle of the AtomGroup: an atom in atomistic
systems, a bead in coarse-grained ones (`cg=True`).

- Every atom is a sphere of its van der Waals radius plus the probe radius
  (default 0.191 nm, a Martini tiny bead), covered with geodesic dots on the sphere.
- A dot is exposed if it lies outside the probe-expanded spheres of all other
  atoms in the respective selection.
- The area of an atom is 4πR² × (exposed dots / all dots).

