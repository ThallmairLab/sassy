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
res = sasa.run(verbose=True)                # calculation with progress bar -> Results
res.total_sasa                                   # (n_frames,) nm²
res.body_sasa, res.body_labels                      # (n_bodies, n_frames), ['GSH1', 'GSH2', ...]
res.select("name S*")                       # (n_frames,) summed over the selection

# a ligand covered by the protein: only the ligand's dots are computed
ligand = SASA(u.select_atoms("resname LIG"), environment="protein", buried=True)
ligand.run().total_sasa         # (n_frames,) SASA of the ligand left uncovered by the protein
ligand.buried.total_sasa        # (n_frames,) area of the ligand buried by the protein
ligand.save("ligand.npz")  # everything, see "Results" below
```
### Command line

```bash
# one column per molecule, each on its own, and the total, as xvg, saves plots in plots directory
sassy -s topol.tpr -f traj.xtc -select "moltype GSH" -columns bodies total -o sasa.xvg -plots plots

# all results (every body, the total, every atom) to analyse later with sassy.load
sassy -s topol.tpr -f traj.xtc -select "moltype GSH" -columns atoms -o sasa.npz

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

With `-buried` (`buried=True`), the SASA is also computed without the environment, and
the area the environment buries (the SASA without minus with it) is available too.

The environment atoms within reach of a body are taken in its periodic image, so the
environment does not need to be whole; the bodies are assumed to be whole. Only the
dots of the selection are computed, so a small ligand in a large protein is fast. The
result equals `-select "protein or resname LIG" -bodies selection -group "resname LIG"`
(and `gmx sasa -surface "Protein|LIG" -output LIG`), which computes the dots of every
protein atom too.

### Results

`SASA.run()` returns a `Results` object (also `sasa.results`). With `buried=True`,
`sasa.free` (without the environment) and `sasa.buried` (the area the environment buries)
are `Results` objects too, with the same attributes and methods:

| | `Results` | `-columns` (command line) |
|---|---|---|
| every body | `.body_sasa` (n_bodies, n_frames), `.body_labels`, `.body('GSH2')` | `bodies` (default) |
| total | `.total_sasa` (n_frames,) | `total` |
| every body type | `.type_sasa()` {type: (n_frames,)}, `.body_type_labels` | `types` |
| every atom | `.atom_sasa` (n_atoms, n_frames), `.atom_labels` (needs `keep_atoms`) | `atoms` |
| a selection of atoms | `.select("name S*")` (needs `keep_atoms`) | `-group "name S*"` |
| times, frames | `.times` (ps), `.frames` | first column of the xvg |

`-columns` takes several levels, in the order given (e.g. `-columns bodies total types`),
and `-group` adds group columns after them; with `-buried` the same columns of the
buried area follow, labelled `<column> buried`. `res.table(["bodies", "total"],
groups=["name S*"])` gives the same table in Python.

`-o x.xvg` writes the table as text. `-o x.npz` (or `sasa.save("x.npz")`) writes all
results, with how they were computed and the table; read them with `sassy.load`:

```python
from sassy import load
saved = load("ligand.npz")          # saved.results, saved.free, saved.buried, saved.meta
saved.buried.type_sasa()
saved.results.select([0, 1, 2])     # atoms by index; by selection string with load("ligand.npz", universe)
```

### Surface dots

The structure of one frame and its solvent accessible surface dots (a Connolly-style
dot surface) can be written for viewing, as PDB or GRO (by the file extension; `-surface-ext`
on the command line): dots are named `DOT` and carry the residue name and id of the atom they
belong to (select them e.g. with `name DOT`). In PDB, atoms are ATOM and dots HETATM records,
and the B-factor of an atom and of its dots is the atom's SASA (nm²), to colour by exposure.
PDB numbers at most 99,999 records (a warning is given above it); use GRO for larger surfaces.

The command line writes one surface per body type, of a representative
body and frame of that type: statistics are taken over the SASA distribution of the
type only (all its bodies in all analysed frames). The statistic is
the median by default, or can be specified using `-surface-stat`: `mean`, `mode` (the most probable
SASA), the quartiles `q1`/`q3`, `min`, `max`, `mean+std`, `mean-std`, a custom function or a fixed SASA
value (in nm²). Instead, a given frame (`-surface-frame`: per type the body closest to the
median in that frame) or the frame with the lowest RMSD to a reference structure (same
atoms; one type or `-surface-body`), superposed onto it.

```python
sasa = SASA(u.select_atoms("moltype GSH"))
sasa.run()
sasa.surface(body_type="GSH").write("GSH_sasa.pdb")        # representative GSH and frame
sasa.surface(statistic="mean+std").write("dots.gro")        # frame closest to mean + std, as GRO
```

```bash
sassy -s topol.tpr -f traj.xtc -select "moltype GSH or moltype SOL" -surface surfaces  					  # surfaces/GSH_sasa.pdb, surfaces/SOL_sasa.pdb
sassy -s topol.tpr -f traj.xtc -select "moltype GSH" -surface -surface-body GSH1 -surface-stat mean   # ./GSH1_sasa.pdb
sassy -s cg.tpr -f cg.xtc -select "moltype GSH" -cg -surface cg -surface-ref mapped.pdb					  # closest to reference structure
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

