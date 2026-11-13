# sassy

Calculates the solvent accessible surface areas (SASA) of molecules and atoms over trajectories. Supports atomistic and coarse-grained (CG) Martini 3 trajectories.
Uses a fast Shrake-Rupley algorithm on a [MDAnalysis](https://www.mdanalysis.org) AtomGroup.

## Installation

```bash
git clone https://github.com/ThallmairLab/sassy
cd sassy
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

# a ligand covered by the protein: only the ligand's dots are computed
ligand = SASA(u.select_atoms("resname LIG"), environment="protein", buried=True)
ligand.run()
ligand.total_sasa          # (n_frames,) SASA of the ligand left uncovered by the protein
ligand.get_buried_sasa()   # (n_frames,) area of the ligand buried by the protein
```
### Command line

```bash
# one column per molecule, each on its own, as xvg, saves plots in plots directory
sassy -s topol.tpr -f traj.xtc -select "moltype GSH" -total -o sasa.xvg -plots plots

# SASA of the whole selection (its molecules bury each other)
sassy -s topol.tpr -f traj.xtc -select "resname GSH" -bodies selection -o sasa.xvg -plots plots

# one column per residue, each on its own
sassy -s topol.tpr -f traj.xtc -select "protein" -bodies residue -o residues.xvg

# a ligand covered by the protein, and the area the protein buries
sassy -s topol.tpr -f traj.xtc -select "resname LIG" -env protein -buried -o ligand.xvg
```

### What buries an atom

Two independent settings decide which atoms bury the surface of an atom:

| setting | values | meaning |
|---|---|---|
| bodies (`-bodies`, `bodies=`) | `molecule` (default) | every molecule on its own: other molecules do not bury it |
| | `residue` | every residue on its own, labelled `<resname><resid>` (e.g. `ALA12`) |
| | `selection` | the whole selection as one body: its molecules bury each other |
| environment (`-env`, `environment=`) | a selection | atoms that bury every body without being counted (e.g. the protein around a ligand); every body is covered by the environment, not by the other bodies |

One column is written per body. `-per-atom`, `-group` and `-total` choose other
columns and work with any setting. With `-buried` (`buried=True`), the area buried by
the environment (the SASA without minus with it) is added as columns labelled
`<column> buried`.

The environment atoms within reach of a body are taken in its periodic image, so the
environment does not need to be whole; the bodies are assumed to be whole. Only the
dots of the selection are computed, so a small ligand in a large protein is fast. The
result equals `-select "protein or resname LIG" -bodies selection -group "resname LIG"`
(and `gmx sasa -surface "Protein|LIG" -output LIG`), which computes the dots of every
protein atom too.

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

Surfaces use fewer dots (`ndots` / `-surface-ndots`, default 240) than the SASA. With an
environment, the dots cover only the part left uncovered, and the environment atoms
within reach are written as chain B.

## Method

As in MDAnalysis, "atom" means every particle of the AtomGroup: an atom in atomistic
systems, a bead in coarse-grained ones (`cg=True`).

- Every atom is a sphere of its van der Waals radius plus the probe radius
  (default 0.191 nm, a Martini tiny bead), covered with geodesic dots on the sphere.
- A dot is exposed if it lies outside the probe-expanded spheres of all other
  atoms of its body and of the environment.
- The area of an atom is 4πR² × (exposed dots / all dots).

