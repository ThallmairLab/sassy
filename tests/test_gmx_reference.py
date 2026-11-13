"""
Comparison with gmx sasa (GROMACS 2024.5) on atomistic GSH, all 501 frames.

The reference was made with the radii of sassy (gmx_vdwradii.dat, copied to
vdwradii.dat in the working directory) and

    gmx sasa -s atomistic.tpr -f atomistic.xtc -surface all -probe 0.191 -ndots 4800 -o gmx_sasa_area.xvg
"""
import numpy as np
import MDAnalysis as mda

from sassy import AA_RADII, SASA
from conftest import DATA


def test_radii_file_matches():
    """The reference used the radii of sassy."""
    radii = {line.split()[1]: float(line.split()[2])
             for line in (DATA / 'gmx_vdwradii.dat').read_text().splitlines()}
    assert radii == AA_RADII


def test_matches_gmx_sasa():
    """
    The SASA of every frame agrees with gmx sasa within the 3 decimals that gmx
    writes (plus float32 rounding of the coordinates).
    """
    reference = np.loadtxt(DATA / 'gmx_sasa_area.xvg', comments=['#', '@'])
    universe = mda.Universe(str(DATA / 'atomistic.tpr'), str(DATA / 'atomistic.xtc'))
    sasa = SASA(universe.atoms).run().bodies[0]
    assert len(sasa) == len(reference) == 501
    assert np.allclose(sasa, reference[:, 1], rtol=0, atol=1e-3)
