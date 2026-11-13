"""Environment: atoms that bury the bodies without being counted (e.g. a protein around a ligand)."""
import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.lib.mdamath import triclinic_vectors
from MDAnalysis.lib.transformations import rotation_matrix

from sassy import SASA, environment_copies
from sassy.bin.sassy import main
from conftest import DATA

TPR, XTC = str(DATA / 'atomistic.tpr'), str(DATA / 'atomistic.xtc')
BOX = np.array([50.0, 50.0, 50.0, 60.0, 60.0, 90.0]) # Å, degrees: triclinic


def copies(gsh, shifts, box=None):
    """
    Universe of GSH copies (molnums 0, 1, ...; resids 1, 2, ...), copy k shifted by ``shifts[k]`` Å
    along x, so that they overlap; with a ``box`` in every frame.
    """
    positions = gsh.universe.trajectory.coordinate_array[:, gsh.indices]
    u = mda.Merge(*[gsh] * len(shifts))
    u.add_TopologyAttr('molnums', np.arange(len(shifts)))
    u.residues.resids = np.arange(1, len(shifts) + 1)
    coordinates = np.concatenate([positions + [shift, 0, 0] for shift in shifts], axis=1)
    if box is None:
        u.load_new(coordinates, order='fac')
    else:
        u.load_new(coordinates, order='fac', dimensions=np.tile(box, (len(coordinates), 1)))
    return u


@pytest.fixture
def complex_(gsh_aa):
    """A 'ligand' (resid 1) between two overlapping copies, the 'protein' (resids 2, 3)."""
    return copies(gsh_aa, [0.0, 4.0, -4.0])


def test_equals_whole_selection(complex_):
    """
    The SASA of the ligand covered by the environment equals its part of the SASA of
    ligand and environment as one body, but only the ligand's dots are computed.
    """
    ligand, protein = complex_.select_atoms('resid 1'), complex_.select_atoms('resid 2 3')
    covered = SASA(ligand, ndots=240, environment=protein)
    per_atom = covered.run().atom_sasa
    whole = SASA(ligand | protein, ndots=240, bodies='selection')
    whole.run()
    assert np.allclose(covered.results.total_sasa, whole.results.select(ligand), rtol=1e-9)
    assert np.allclose(per_atom, whole.results.select(ligand, per_atom=True), rtol=1e-9, atol=1e-12)
    alone = SASA(ligand, ndots=240).run().body_sasa[0]
    assert np.all(covered.results.total_sasa < 0.9 * alone) # buried by the environment
    # blocks of one frame: the same
    one_by_one = SASA(ligand, ndots=240, environment=protein, max_frame_atoms=1).run().body_sasa
    assert np.allclose(one_by_one, covered.results.body_sasa, rtol=1e-9)
    # a selection string and an AtomGroup give the same
    assert np.array_equal(SASA(ligand, ndots=240, environment='resid 2 3').run().body_sasa, covered.results.body_sasa)
    # far away: no effect
    far = copies(complex_.atoms[:36], [0.0, 100.0])
    assert np.allclose(SASA(far.select_atoms('resid 1'), ndots=240, environment='resid 2').run().body_sasa[0], alone)


def test_periodic_images(gsh_aa):
    """
    The environment is taken in the periodic image of the body (triclinic box): moving the
    ligand or single environment atoms by box vectors does not change the SASA.
    """
    reference = copies(gsh_aa, [0.0, 4.0, -4.0], box=BOX)
    expected = SASA(reference.select_atoms('resid 1'), ndots=240, environment='resid 2 3').run().body_sasa
    vectors = triclinic_vectors(BOX)
    moved = copies(gsh_aa, [0.0, 4.0, -4.0], box=BOX)
    coordinates = moved.trajectory.coordinate_array # (n_frames, n_atoms, 3), in memory
    ligand, environment = moved.select_atoms('resid 1').indices, moved.select_atoms('resid 2 3').indices
    coordinates[:, ligand] += vectors[0] - vectors[2] # the whole ligand
    rng = np.random.default_rng(0)
    shifts = rng.integers(-1, 2, size=(len(environment), 3)) @ vectors # every environment atom on its own
    coordinates[:, environment] += shifts
    result = SASA(moved.select_atoms('resid 1'), ndots=240, environment='resid 2 3').run().body_sasa
    assert np.allclose(result, expected, rtol=1e-6)
    # without the box, the broken environment would bury less
    assert np.allclose(expected, SASA(copies(gsh_aa, [0.0, 4.0, -4.0]).select_atoms('resid 1'), ndots=240,
                                      environment='resid 2 3').run().body_sasa, rtol=1e-6)


def test_environment_copies():
    """Environment atoms within the cutoff of every body, once per body, in the image of the body."""
    xyz = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]])
    body = np.array([0, 0, 1])
    env = np.array([[0.3, 0.0, 0.0], [2.2, 0.0, 0.0], [1.0, 0.0, 0.0], [4.9, 0.0, 0.0]])
    bodies, index, positions = environment_copies(xyz, body, env, 0.5)
    assert sorted(zip(bodies.tolist(), index.tolist())) == [(0, 0), (1, 1)]
    box = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0]) # nm
    bodies, index, positions = environment_copies(xyz, body, env, 0.5, box)
    assert sorted(zip(bodies.tolist(), index.tolist())) == [(0, 0), (0, 3), (1, 1)]
    assert np.allclose(positions[index == 3], [[-0.1, 0.0, 0.0]], atol=1e-6) # the image next to body 0
    empty = environment_copies(xyz, body, np.zeros((0, 3)), 0.5)
    assert all(len(a) == 0 for a in empty)


def test_molecule_bodies(gsh_aa):
    """
    Per molecule, every ligand is covered by the environment, but not by the other ligands;
    as one body, the ligands also bury each other.
    """
    u = copies(gsh_aa, [0.0, 4.0, -4.0])
    ligands, protein = u.select_atoms('resid 1 2'), u.select_atoms('resid 3')
    covered = SASA(ligands, ndots=240, environment=protein)
    covered.run(stop=5)
    for k, resid in enumerate((1, 2)):
        alone = SASA(u.select_atoms(f'resid {resid}'), ndots=240, environment=protein).run(stop=5).body_sasa[0]
        assert np.allclose(covered.results.body_sasa[k], alone, rtol=1e-9)
    together = SASA(ligands, ndots=240, environment=protein, bodies='selection')
    together.run(stop=5)
    assert np.all(together.results.total_sasa < covered.results.total_sasa)
    assert 'covered by an environment of 36 atom(s)' in str(covered)


def test_buried(complex_):
    """With buried, also the SASA without the environment, and the area buried by it."""
    ligand = complex_.select_atoms('resid 1')
    sasa = SASA(ligand, ndots=240, environment='resid 2 3', buried=True)
    sasa.run(stop=6)
    free = SASA(ligand, ndots=240)
    free.run(stop=6)
    assert np.allclose(sasa.free.body_sasa, free.results.body_sasa, rtol=1e-9)
    assert np.allclose(sasa.free.atom_sasa, free.results.atom_sasa, rtol=1e-9)
    buried = sasa.buried.total_sasa
    assert np.allclose(buried, free.results.total_sasa - sasa.results.total_sasa)
    assert np.all(buried > 0)
    assert np.all(sasa.buried.atom_sasa >= -1e-12) # an atom is never more exposed with it
    assert np.allclose(sasa.buried.select('name S*'), free.results.select('name S*') - sasa.results.select('name S*'))
    assert np.allclose(sasa.buried.body('GSH1'), buried)
    assert sasa.buried.body_sasa.shape == (1, 6)
    # without buried: no free SASA
    plain = SASA(ligand, ndots=240, environment='resid 2 3')
    plain.run(stop=2)
    assert plain.free is None and plain.buried is None


def test_environment_errors(complex_, gsh_aa):
    ligand = complex_.select_atoms('resid 1')
    with pytest.raises(ValueError, match='both in the AtomGroup and in the environment'):
        SASA(ligand, environment='resid 1 2')
    with pytest.raises(ValueError, match='no atoms'):
        SASA(ligand, environment='resid 99')
    with pytest.raises(ValueError, match='different Universe'):
        SASA(ligand, environment=gsh_aa)
    with pytest.raises(TypeError, match='selection string or an AtomGroup'):
        SASA(ligand, environment=3)
    with pytest.raises(ValueError, match='buried needs an environment'):
        SASA(ligand, buried=True)


def test_surface_with_environment(complex_, tmp_path):
    """
    The surface of a covered ligand: dots only on the part left uncovered, the area equal
    to its SASA, the environment atoms within reach as chain B; superposed with a reference.
    """
    ligand = complex_.select_atoms('resid 1')
    sasa = SASA(ligand, ndots=240, environment='resid 2 3')
    sasa.run()
    out = tmp_path / 'ligand.pdb'
    surface = sasa.surface(frame=3)
    surface.write(out)
    assert surface.area == pytest.approx(sasa.results.body_sasa[0, 3], rel=1e-6)
    assert np.all(surface.owners < ligand.n_atoms)
    assert 0 < surface.environment.n_atoms <= 72
    assert any('Covered by an environment of 72 atoms' in remark for remark in surface.remarks)
    with pytest.warns(UserWarning):
        written = mda.Universe(str(out))
    assert written.select_atoms('chainID B').n_atoms == len(np.unique(surface.environment.indices))
    assert written.select_atoms('chainID A').n_atoms == ligand.n_atoms

    # the reference: the ligand of frame 3, rotated and moved; the environment follows the superposition
    complex_.trajectory[3]
    rotation = rotation_matrix(0.7, [1, 2, 3])[:3, :3]
    reference = (ligand.positions - ligand.center_of_geometry()) @ rotation.T + [5.0, -3.0, 2.0]
    aligned = sasa.surface(reference=reference / 10)
    assert aligned.frame == 3

    def distances(s):
        return np.linalg.norm(s.positions[:, None] - s.environment_positions[None], axis=-1)
    assert np.allclose(distances(aligned), distances(surface), atol=1e-4)
    assert aligned.area == pytest.approx(surface.area, rel=0.03) # other orientation of the dot sphere


@pytest.mark.filterwarnings('ignore::UserWarning') # MDAnalysis writing and reading the PDB
def test_cli_environment(complex_, tmp_path, capsys):
    """-env and -buried: the covered SASA columns, then the same columns of the buried area."""
    pdb = str(tmp_path / 'complex.pdb')
    complex_.trajectory[0]
    complex_.atoms.write(pdb)
    out = tmp_path / 'ligand.xvg'
    assert main(['-s', pdb, '-select', 'resid 1', '-env', 'resid 2 3', '-buried', '-columns', 'bodies', 'total', '-ndots', '240',
                 '-q', '-o', str(out)]) == 0
    data = np.loadtxt(out, comments=['#', '@'])
    legends = [line.split('"')[1] for line in open(out) if line.startswith('@ s')]
    assert legends == ['GSH1', 'total', 'GSH1 buried', 'total buried']
    u = mda.Universe(pdb)
    sasa = SASA(u.select_atoms('resid 1'), ndots=240, environment='resid 2 3', buried=True)
    sasa.run()
    assert data[1] == pytest.approx(sasa.results.total_sasa[0], abs=1e-4)
    assert data[3] == pytest.approx(sasa.buried.total_sasa[0], abs=1e-4)

    # groups and npz; plots and surfaces with the environment
    out = tmp_path / 'ligand.npz'
    assert main(['-s', pdb, '-select', 'resid 1', '-env', 'resid 2 3', '-buried', '-group', 'name S*',
                 '-ndots', '240', '-q', '-o', str(out), '-plots', str(tmp_path / 'plots'),
                 '-surface', str(tmp_path / 'surf')]) == 0
    from sassy import load
    result = np.load(out)
    assert result['column_labels'].tolist() == ['name S*', 'name S* buried']
    saved = load(out)
    assert np.allclose(saved.buried.total_sasa, sasa.buried.total_sasa, atol=1e-12)
    assert saved.meta['environment'] == 'resid 2 3' and saved.meta['n_environment'] == 72
    assert 'ATOM' in (tmp_path / 'surf' / 'GSH_sasa.pdb').read_text()
    assert (tmp_path / 'plots' / 'sasa_total.png').is_file()

    with pytest.raises(SystemExit): # -buried needs -env
        main(['-s', pdb, '-select', 'resid 1', '-buried'])
    capsys.readouterr()
    assert main(['-s', pdb, '-select', 'resid 1', '-env', 'resid 1 2', '-q', '-o', str(tmp_path / 'x.xvg')]) == 1
    assert 'both in the AtomGroup and in the environment' in capsys.readouterr().err
    assert main(['-s', pdb, '-select', 'resid 1', '-env', 'resid 99', '-q', '-o', str(tmp_path / 'x.xvg')]) == 1
    assert 'the environment' in capsys.readouterr().err
