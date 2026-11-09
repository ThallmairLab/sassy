"""Command line interface."""
import numpy as np
import pytest
import MDAnalysis as mda

from fastsasa import SASA
from fastsasa.bin.fastsasa import main
from conftest import DATA

TPR, XTC = str(DATA / 'atomistic.tpr'), str(DATA / 'atomistic.xtc')


def read_xvg(path):
    legends = [line.split('"')[1] for line in open(path) if line.startswith('@ s')]
    return np.loadtxt(path, comments=['#', '@']), legends


def test_cli_molecules(tmp_path):
    """The xvg has the time and one column per molecule, equal to SASA.run()."""
    out = tmp_path / 'sasa.xvg'
    assert main(['-s', TPR, '-f', XTC, '-select', 'moltype GSH', '-ndots', '240',
                 '-skip', '25', '-total', '-q', '-o', str(out)]) == 0
    data, legends = read_xvg(out)
    universe = mda.Universe(TPR, XTC)
    sasa = SASA(universe.select_atoms('moltype GSH'), ndots=240)
    expected = sasa.run(step=25)
    assert legends == ['GSH1', 'total']
    assert np.allclose(data[:, 0], sasa.times, atol=1e-3)
    assert np.allclose(data[:, 1], expected[0], atol=1e-4)
    assert np.allclose(data[:, 2], expected.sum(axis=0), atol=1e-4)


def test_cli_groups_atoms_npz(tmp_path):
    """Groups, per-atom columns and npz output."""
    out = tmp_path / 'groups.npz'
    assert main(['-s', TPR, '-f', XTC, '-select', 'moltype GSH', '-ndots', '240', '-e', '3',
                 '-group', 'name S*', '-group', 'name C*', '-q', '-o', str(out)]) == 0
    result = np.load(out)
    universe = mda.Universe(TPR, XTC)
    sasa = SASA(universe.select_atoms('moltype GSH'), ndots=240)
    sasa.run(stop=3)
    assert result['labels'].tolist() == ['name S*', 'name C*']
    assert result['frames'].tolist() == [0, 1, 2]
    assert np.allclose(result['sasa'], [sasa.get_sasa('name S*'), sasa.get_sasa('name C*')])

    out = tmp_path / 'atoms.xvg'
    assert main(['-s', TPR, '-f', XTC, '-select', 'moltype GSH', '-ndots', '240', '-e', '2',
                 '-per-atom', '-radius', 'S=0.18', '-q', '-o', str(out)]) == 0
    data, legends = read_xvg(out)
    assert data.shape == (2, 1 + sasa.ag.n_atoms)
    assert legends[0].startswith('GSH1:')


def test_cli_errors(tmp_path, capsys):
    """Empty selections and conflicting or malformed options are reported."""
    with pytest.warns(UserWarning, match='No coordinate reader'):
        assert main(['-s', TPR, '-q', '-o', str(tmp_path / 'x.xvg')]) == 1
    assert 'give a trajectory with -f' in capsys.readouterr().err
    assert main(['-s', TPR, '-f', XTC, '-select', 'name XYZ', '-q', '-o', str(tmp_path / 'x.xvg')]) == 1
    assert 'contains no atoms' in capsys.readouterr().err
    assert main(['-s', TPR, '-f', XTC, '-group', 'name XYZ', '-q', '-o', str(tmp_path / 'x.xvg')]) == 1
    with pytest.raises(SystemExit):
        main(['-s', TPR, '-per-atom', '-group', 'all'])
    with pytest.raises(SystemExit):
        main(['-s', TPR, '-radius', 'Na0.2'])


def two_gsh(tmp_path, shift):
    """pdb (no molnums) of two GSH copies, the second shifted by ``shift`` Å along x."""
    gsh = mda.Universe(TPR, XTC).select_atoms('moltype GSH')
    two = mda.Merge(gsh, gsh)
    two.atoms[gsh.n_atoms:].positions += [shift, 0, 0]
    path = str(tmp_path / 'two.pdb')
    two.atoms.write(path)
    return path


def test_cli_whole_selection(tmp_path):
    """
    By default every molecule is on its own; with -whole the selection is one body
    (one column, molecules bury each other).
    """
    pdb = two_gsh(tmp_path, 4.0) # overlapping copies
    whole, single = tmp_path / 'whole.xvg', tmp_path / 'single.xvg'
    assert main(['-s', pdb, '-ndots', '240', '-whole', '-q', '-o', str(whole)]) == 0
    assert main(['-s', pdb, '-ndots', '240', '-total', '-q', '-o', str(single)]) == 0
    whole, legends = read_xvg(whole)
    assert legends == ['all']
    single, legends = read_xvg(single)
    assert legends == ['GSH1', 'GSH2', 'total']
    assert single[1] == pytest.approx(single[2])
    assert whole[1] < 0.9 * single[3] # buried by each other

    far = two_gsh(tmp_path, 100.0) # far apart: no difference
    out = tmp_path / 'far.xvg'
    assert main(['-s', far, '-ndots', '240', '-whole', '-q', '-o', str(out)]) == 0
    assert read_xvg(out)[0][1] == pytest.approx(single[3], abs=1e-3)


def test_cli_molecule_types_warning(tmp_path, capsys):
    """Per molecule (default) warns about a selection of several molecule types, -whole does not."""
    from fastsasa import n_molecule_types
    gsh = mda.Universe(TPR, XTC).select_atoms('moltype GSH')
    assert n_molecule_types(gsh) == 1
    two = mda.Merge(gsh, gsh)
    two.add_TopologyAttr('molnums', [0, 1])
    two.add_TopologyAttr('moltypes', ['GSH', 'GSH'])
    assert n_molecule_types(two.atoms) == 1
    two.residues[1].moltype = 'OTHER'
    assert n_molecule_types(two.atoms) == 2
    two.del_TopologyAttr('moltypes') # without moltypes: by atom and residue names
    assert n_molecule_types(two.atoms) == 1
    two.residues[1].resname = 'OTH'
    assert n_molecule_types(two.atoms) == 2

    import fastsasa.bin.fastsasa as cli
    out = str(tmp_path / 'x.xvg')
    original = cli.n_molecule_types
    cli.n_molecule_types = lambda ag: 2
    try:
        assert main(['-s', TPR, '-f', XTC, '-e', '1', '-ndots', '240', '-whole', '-q', '-o', out]) == 0
        assert 'WARNING' not in capsys.readouterr().err
        assert main(['-s', TPR, '-f', XTC, '-e', '1', '-ndots', '240', '-q', '-o', out]) == 0
        assert '2 different molecule types' in capsys.readouterr().err
    finally:
        cli.n_molecule_types = original


def test_cli_logging(tmp_path, capsys):
    """-q: nothing on stderr, default: info, -v: debug; -v and -q exclude each other."""
    out = str(tmp_path / 'x.xvg')
    run = ['-s', TPR, '-f', XTC, '-e', '2', '-ndots', '240', '-o', out]
    assert main(run + ['-q']) == 0
    assert capsys.readouterr().err == ''
    assert main(run) == 0
    err = capsys.readouterr().err
    assert '36 atoms in 1 molecule(s), 2 frame(s)' in err and 'wrote 1 column(s)' in err
    assert 'DEBUG' not in err
    assert main(run + ['-v']) == 0
    err = capsys.readouterr().err
    assert 'DEBUG   [fastsasa.groups] molecules from the molnums' in err
    assert 'radii (nm) used' in err and 'frames/s' in err
    with pytest.raises(SystemExit):
        main(run + ['-v', '-q'])
    # errors of the SASA setup are logged, not raised
    assert main(run + ['-q', '-radius', 'C=-1']) == 1
    assert 'fastsasa: ERROR: Radii must be >= 0' in capsys.readouterr().err
