from pathlib import Path
import pytest
import MDAnalysis as mda

DATA = Path(__file__).parent / 'data'


@pytest.fixture
def gsh_aa():
    """
    Atomistic glutathione (GSH, a single molecule), every 25th frame (20 frames).
    """
    universe = mda.Universe(str(DATA / 'atomistic.tpr'), str(DATA / 'atomistic.xtc'),
                            in_memory=True, in_memory_step=25)
    return universe.select_atoms('moltype GSH')
