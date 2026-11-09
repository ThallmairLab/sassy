"""SASA of molecules and atoms over a trajectory."""
import numpy as np
import pytest

from fastsasa import SASA, atom_radii, atom_symbols, split_molecules, surface_dots


def test_sasa_class(gsh_aa):
    """
    The SASA of every frame equals the area of the surface dots of that frame,
    independent of how many frames are processed at once.
    """
    sasa = SASA(gsh_aa, ndots=240).run()
    assert sasa.shape == (1, gsh_aa.universe.trajectory.n_frames)
    radii = atom_radii(gsh_aa)
    for frame in (0, 7, 19):
        gsh_aa.universe.trajectory[frame]
        _, _, area = surface_dots(gsh_aa.positions / 10, radii, ndots=240)
        assert sasa[0, frame] == pytest.approx(area, rel=1e-6)
    # one frame per block
    one_by_one = SASA(gsh_aa, ndots=240, max_frame_atoms=1).run()
    assert np.allclose(one_by_one, sasa, rtol=1e-6)
    assert np.allclose(SASA(gsh_aa, ndots=240).run(start=5, step=3), sasa[:, 5::3], rtol=1e-6)


def test_sasa_per_atom(gsh_aa):
    """
    Atom SASAs are in the order of the AtomGroup, sum to the molecule SASA, equal the
    dots of every atom of surface_dots, are 0 for atoms with radius 0, and do not
    depend on how many frames are processed at once.
    """
    radii = atom_radii(gsh_aa)
    radii[3] = 0
    sasa = SASA(gsh_aa, radii=radii, ndots=240)
    per_atom = sasa.run(per_atom=True)
    n_frames = gsh_aa.universe.trajectory.n_frames
    assert per_atom.shape == (gsh_aa.n_atoms, n_frames)
    assert np.all(per_atom[3] == 0)
    assert np.allclose(per_atom.sum(axis=0), sasa.run()[0], rtol=1e-10)
    gsh_aa.universe.trajectory[7]
    _, owners, _ = surface_dots(gsh_aa.positions / 10, radii, ndots=240)
    ext = radii + 0.191
    expected = np.bincount(owners, minlength=gsh_aa.n_atoms) * 4 * np.pi * ext**2 / len(sasa.unit)
    assert np.allclose(per_atom[:, 7], expected, rtol=1e-6)
    one_by_one = SASA(gsh_aa, radii=radii, ndots=240, max_frame_atoms=1).run(per_atom=True, start=2, step=3)
    assert np.allclose(one_by_one, per_atom[:, 2::3], rtol=1e-6)


def test_sasa_progress(gsh_aa, capsys):
    """
    The progress bar goes to stderr, counts all frames, and does not change the result.
    """
    sasa = SASA(gsh_aa, ndots=240, max_frame_atoms=200)
    quiet = sasa.run(step=2)
    assert capsys.readouterr().err == ''
    assert np.array_equal(sasa.run(step=2, verbose=True), quiet)
    n_frames = len(gsh_aa.universe.trajectory[::2])
    assert f'{n_frames}/{n_frames}' in capsys.readouterr().err
    assert np.allclose(sasa.run(step=2, per_atom=True, verbose=True).sum(axis=0), quiet[0], rtol=1e-10)


def test_sasa_getters(gsh_aa):
    """
    run() stores the atom SASA; get_sasa sums any selection (string, AtomGroup,
    residue, atom) and get_molecule_sasa returns molecules by index.
    """
    import MDAnalysis as mda
    positions = gsh_aa.universe.trajectory.coordinate_array[:, gsh_aa.indices]
    two = mda.Merge(gsh_aa, gsh_aa)
    two.add_TopologyAttr('molnums', [0, 1])
    two.load_new(np.concatenate([positions, positions + 50.0], axis=1), order='fac')
    sasa = SASA(two.atoms, ndots=240)
    with pytest.raises(RuntimeError, match='run'):
        sasa.get_sasa()
    molecules = sasa.run(step=4)
    n_frames = len(two.trajectory[::4])
    assert sasa.sasa.shape == (two.atoms.n_atoms, n_frames)
    assert sasa.frames.tolist() == list(range(0, two.trajectory.n_frames, 4))
    assert np.allclose(sasa.times, [ts.time for ts in two.trajectory[::4]])

    # molecules by index
    assert np.array_equal(sasa.get_molecule_sasa(), molecules)
    assert np.array_equal(sasa.get_molecule_sasa(1), molecules[1])
    assert np.array_equal(sasa.get_molecule_sasa(-1), molecules[1])
    assert np.array_equal(sasa.get_molecule_sasa([1, 0]), molecules[[1, 0]])
    with pytest.raises(IndexError):
        sasa.get_molecule_sasa(2)

    # selections: all atoms, a molecule, a string, a residue, a single atom
    n = gsh_aa.n_atoms
    assert np.allclose(sasa.get_sasa(), molecules.sum(axis=0))
    assert np.allclose(sasa.get_sasa(two.atoms[n:]), molecules[1])
    sulfur = sasa.get_sasa('name S*', per_atom=True)
    assert sulfur.shape == (len(two.select_atoms('name S*')), n_frames)
    assert np.allclose(sasa.get_sasa('name S*'), sulfur.sum(axis=0))
    assert np.allclose(sasa.get_sasa(two.residues[0]), molecules[0])
    assert np.array_equal(sasa.get_sasa(two.atoms[5], per_atom=True), sasa.sasa[[5]])
    # order of the selection is kept
    assert np.array_equal(sasa.get_sasa(two.atoms[[7, 2]], per_atom=True), sasa.sasa[[7, 2]])

    # selections outside the AtomGroup are rejected
    part = SASA(two.atoms[:n], ndots=240)
    part.run(stop=2)
    with pytest.raises(ValueError, match='not in the AtomGroup'):
        part.get_sasa(two.atoms[n:])
    with pytest.raises(ValueError, match='different Universe'):
        part.get_sasa(gsh_aa)


def test_sasa_zero_radius(gsh_aa):
    """
    Atoms with radius 0 are ignored: giving an atom radius 0 equals removing it.
    """
    radii = atom_radii(gsh_aa)
    radii[3] = 0
    without = SASA(gsh_aa, radii=radii, ndots=240).run()
    removed = gsh_aa[np.arange(gsh_aa.n_atoms) != 3]
    assert np.allclose(without, SASA(removed, ndots=240).run(), rtol=1e-6)


def test_unknown_element(gsh_aa):
    """
    An atom whose element has no atomistic radius raises an error naming the atom.
    """
    atoms = gsh_aa.universe.copy().select_atoms('moltype GSH')
    atoms[0].element = 'Na'
    with pytest.raises(ValueError, match=rf"{atoms[0].resname}{atoms[0].resid}:{atoms[0].name} \(element 'Na'\)"):
        atom_radii(atoms)
    with pytest.raises(ValueError, match='No atomistic SASA radius known for 1 atom'):
        SASA(atoms)


def test_radii_dict(gsh_aa):
    """
    A dict of radii by element adds missing elements and overrides defaults;
    SASA(radii=dict) equals SASA with the resulting per-atom radii.
    """
    atoms = gsh_aa.universe.copy().select_atoms('moltype GSH')
    atoms[0].element = 'Na'
    radii = atom_radii(atoms, table={'Na': 0.227, 'C': 0.170})
    expected = atom_radii(gsh_aa)
    carbon = np.array([el == 'C' for el in atoms.elements])
    expected[carbon] = 0.170
    expected[0] = 0.227
    assert np.allclose(radii, expected)
    by_dict = SASA(atoms, radii={'Na': 0.227, 'C': 0.170}, ndots=240).run()
    assert np.allclose(by_dict, SASA(atoms, radii=radii, ndots=240).run(), rtol=1e-6)
    with pytest.raises(ValueError, match='>= 0'):
        atom_radii(atoms, table={'Na': -0.1})
    with pytest.raises(ValueError, match='one value per atom'):
        SASA(atoms, radii=radii[:-1])
    # keys are case-insensitive
    assert np.allclose(atom_radii(atoms, table={'NA': 0.227, 'c': 0.170}), radii)


def test_guessed_elements(gsh_aa):
    """
    Without elements in the topology, they are guessed from the atom names
    (case-insensitive lookup); empty elements in a topology with elements are not guessed.
    """
    import MDAnalysis as mda
    names = ['CLA', 'SOD', 'NA+', 'OW', 'HW1', 'C1']
    universe = mda.Universe.empty(len(names), trajectory=True)
    universe.add_TopologyAttr('names', names)
    assert atom_symbols(universe.atoms) == ['Cl', 'Na', 'Na', 'O', 'H', 'C']
    assert np.allclose(atom_radii(universe.atoms, table={'na': 0.227}),
                       [0.174, 0.227, 0.227, 0.156, 0.109, 0.175])
    # without elements the guess equals the elements of the tpr
    atoms = gsh_aa.universe.copy().select_atoms('moltype GSH')
    no_elements = mda.Universe.empty(atoms.n_atoms, trajectory=True)
    no_elements.add_TopologyAttr('names', atoms.names)
    assert atom_symbols(no_elements.atoms) == atom_symbols(atoms)
    # an empty element (e.g. a virtual site in a tpr) uses the first letter, not the guess
    atoms[0].element = ''
    atoms[0].name = 'MN1'
    assert atom_symbols(atoms)[0] == 'M'


def test_sasa_per_copy(gsh_aa):
    """
    With several copies, every copy gets its own SASA (copies do not bury each
    other, even when they overlap).
    """
    import MDAnalysis as mda
    n_frames = gsh_aa.universe.trajectory.n_frames
    # two copies of GSH on top of each other, the second shifted by 1 Å
    positions = gsh_aa.universe.trajectory.coordinate_array[:, gsh_aa.indices]
    two = mda.Merge(gsh_aa, gsh_aa)
    two.add_TopologyAttr('molnums', [0, 1])  # one residue per copy
    two.load_new(np.concatenate([positions, positions + 1.0], axis=1), order='fac')
    copies = split_molecules(two.atoms)
    assert [len(copy) for copy in copies] == [gsh_aa.n_atoms] * 2
    assert np.all(copies[0].indices == np.arange(gsh_aa.n_atoms))

    single = SASA(gsh_aa).run()[0]
    per_copy = SASA(two.atoms).run()
    assert per_copy.shape == (2, n_frames)
    # translation does not change the SASA (up to float32 rounding); burial by the
    # overlapping copy would change it by several nm^2
    assert np.allclose(per_copy, single[None], atol=2e-3)


def _topology(names, resnames, resids):
    """Universe with one atom per entry; atoms with the same consecutive resid share a residue."""
    import MDAnalysis as mda
    resids = np.asarray(resids)
    resindex = np.concatenate([[0], np.cumsum(resids[1:] != resids[:-1])])
    first = np.flatnonzero(np.r_[True, resids[1:] != resids[:-1]])
    universe = mda.Universe.empty(len(names), n_residues=len(first), atom_resindex=resindex, trajectory=True)
    universe.add_TopologyAttr('names', names)
    universe.add_TopologyAttr('resnames', np.asarray(resnames)[first])
    universe.add_TopologyAttr('resids', resids[first])
    return universe


@pytest.mark.parametrize('resids, expected', [
    # three copies of a dipeptide, resids continuing or restarting
    ([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
    ([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
    # the offset may differ between copies
    ([1, 1, 2, 2, 10, 10, 11, 11, 3, 3, 4, 4], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
    # resids that do not repeat with an offset: one molecule
    ([1, 1, 2, 2, 3, 3, 5, 5, 5, 5, 6, 6], [0] * 12),
])
def test_molecule_copies(resids, expected):
    """
    Without molnums, molecules are the shortest repeating unit of atom names,
    residue names and resids (up to an offset).
    """
    from fastsasa import copy_length, molecule_index
    universe = _topology(['N', 'C'] * 6, ['ALA', 'ALA', 'GLY', 'GLY'] * 3, resids)
    assert molecule_index(universe.atoms).tolist() == expected
    assert copy_length(universe.atoms) == expected.count(0)


def test_molecule_copies_special():
    """
    Single residues repeat on their own, a non-repeating group is one molecule,
    differing atom names prevent copies, and molnums take precedence.
    """
    from fastsasa import molecule_index
    water = _topology(['OW', 'HW1', 'HW2'] * 4, ['SOL'] * 12, np.repeat([1, 2, 3, 4], 3))
    assert molecule_index(water.atoms).tolist() == np.repeat([0, 1, 2, 3], 3).tolist()
    protein = _topology(['N', 'CA', 'N', 'CA', 'N', 'CB'], ['ALA', 'ALA', 'ALA', 'ALA', 'SER', 'SER'],
                        [1, 1, 2, 2, 3, 3])
    assert molecule_index(protein.atoms).tolist() == [0] * 6
    renamed = _topology(['N', 'C', 'N', 'CX'], ['ALA'] * 4, [1, 1, 2, 2])
    assert molecule_index(renamed.atoms).tolist() == [0] * 4
    water.add_TopologyAttr('molnums', [5, 5, 3, 3])
    assert molecule_index(water.atoms).tolist() == np.repeat([0, 0, 1, 1], 3).tolist()
    assert molecule_index(water.atoms[:0]).tolist() == []


def test_sasa_copies_without_molnums(gsh_aa):
    """
    Two copies of GSH without molnums are detected as two molecules, as with molnums.
    """
    import MDAnalysis as mda
    positions = gsh_aa.universe.trajectory.coordinate_array[:, gsh_aa.indices]
    two = mda.Merge(gsh_aa, gsh_aa)
    two.load_new(np.concatenate([positions, positions + 1.0], axis=1), order='fac')
    if hasattr(two.atoms, 'molnums'):
        two.del_TopologyAttr('molnums')
    assert [len(copy) for copy in split_molecules(two.atoms)] == [gsh_aa.n_atoms] * 2
    with_molnums = SASA(two.atoms, ndots=240).run()
    two.add_TopologyAttr('molnums', [0, 1])
    assert np.allclose(with_molnums, SASA(two.atoms, ndots=240).run())


def test_cg_radii():
    """
    CG radii come from the bead size, the first letter of the bead type: tiny,
    small, dummy, or regular for everything else. Dummy beads have no SASA.
    """
    import MDAnalysis as mda
    from fastsasa import CG_RADII, atom_symbols
    universe = mda.Universe.empty(5, n_residues=1, atom_resindex=[0] * 5, trajectory=True)
    universe.add_TopologyAttr('types', ['TC6', 'SQ5n', 'P2', 'Q4p', 'U'])
    universe.add_TopologyAttr('names', ['A', 'B', 'C', 'D', 'E'])
    universe.add_TopologyAttr('resnames', ['MOL'])
    assert atom_symbols(universe.atoms, cg=True) == ['T', 'S', 'R', 'R', 'U']
    radii = atom_radii(universe.atoms, cg=True)
    assert radii.tolist() == [CG_RADII[s] for s in 'TSRRU']
    # beads far apart: every bead keeps its full sphere, the dummy has none
    universe.atoms.positions = np.arange(5)[:, None] * np.array([[20.0, 0.0, 0.0]])
    per_atom = SASA(universe.atoms, cg=True, ndots=240).run(per_atom=True)
    expected = [4 * np.pi * (r + 0.191)**2 if r else 0.0 for r in radii]
    assert np.allclose(per_atom[:, 0], expected)


def test_save_atoms(gsh_aa, tmp_path):
    """
    Molecule and total SASA are always stored; without save_atoms the per-atom SASA is not,
    and what needs it raises.
    """
    full = SASA(gsh_aa, ndots=240)
    molecules = full.run(step=4)
    assert np.array_equal(full.molecule_sasa, molecules)
    assert np.allclose(full.total_sasa, full.sasa.sum(axis=0))
    light = SASA(gsh_aa, ndots=240, save_atoms=False)
    assert np.array_equal(light.run(step=4), molecules)
    assert light.sasa is None
    assert np.array_equal(light.get_sasa(), full.get_sasa())
    assert np.array_equal(light.get_molecule_sasa(0), molecules[0])
    assert light.write_surface(tmp_path / 'a.pdb').frame == full.write_surface(tmp_path / 'b.pdb').frame
    with pytest.raises(RuntimeError, match='save_atoms'):
        light.get_sasa('name S*')
    with pytest.raises(RuntimeError, match='save_atoms'):
        light.get_sasa(per_atom=True)
    with pytest.raises(ValueError, match='save_atoms'):
        light.run(per_atom=True)
