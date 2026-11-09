"""Surface dots (Connolly-style dot surface) of one frame, written as PDB."""
import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.lib.transformations import rotation_matrix

from fastsasa import SASA, Surface, best_matching_frame, closest_to_median, generate_surface, to_nm
from fastsasa.bin.fastsasa import main
from conftest import DATA

TPR, XTC = str(DATA / 'atomistic.tpr'), str(DATA / 'atomistic.xtc')


def two_molecules(gsh_aa, shift):
    """Two GSH copies (molnums 0, 1), the second shifted by ``shift`` Å along x."""
    positions = gsh_aa.universe.trajectory.coordinate_array[:, gsh_aa.indices]
    two = mda.Merge(gsh_aa, gsh_aa)
    two.add_TopologyAttr('molnums', [0, 1])
    two.load_new(np.concatenate([positions, positions + [shift, 0, 0]], axis=1), order='fac')
    return two.atoms


def test_helpers():
    assert closest_to_median([5.0, 1.0, 3.2, 9.0, 2.9]) == 2
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(10, 3))
    rotation = rotation_matrix(0.7, [1, 2, 3])[:3, :3]
    frames = [rng.normal(size=(10, 3)), reference @ rotation.T + 5.0, rng.normal(size=(10, 3))]
    frame, aligned, rmsd = best_matching_frame(np.array(frames), reference)
    assert frame == 1 and rmsd == pytest.approx(0, abs=1e-6)
    assert np.allclose(aligned, reference, atol=1e-6)


def test_surface_function(gsh_aa, tmp_path):
    """
    Without a SASA: the current frame by default, default or custom radii; the same
    surface as SASA.write_surface.
    """
    gsh_aa.universe.trajectory[5]
    result = generate_surface(gsh_aa)
    assert result.frame == 5 and gsh_aa.universe.trajectory.ts.frame == 5
    assert result.area == pytest.approx(SASA(gsh_aa, ndots=240).write_surface(tmp_path / 'x.pdb', frame=5).area, rel=1e-12)
    assert not any('Custom' in remark for remark in result.remarks)
    sasa = SASA(gsh_aa, ndots=240, radii={'S': 0.2})
    custom = sasa.write_surface(tmp_path / 'x.pdb', frame=5)
    assert 'Custom radii' in custom.remarks
    assert custom.area == pytest.approx(generate_surface(gsh_aa, radii=sasa.radii, frame=5).area)
    assert custom.area != pytest.approx(result.area)
    assert np.array_equal(sasa.molecule_atoms(0).indices, gsh_aa.indices)


def test_surface_area_and_frame(gsh_aa, tmp_path):
    """
    The area of the dots equals the SASA of that frame; by default the frame of the
    run whose SASA is closest to the median is used.
    """
    sasa = SASA(gsh_aa, ndots=240)
    with pytest.raises(RuntimeError, match='run'):
        sasa.write_surface(tmp_path / 'x.pdb')
    series = sasa.run(step=2)[0]
    surface = sasa.write_surface(tmp_path / 'x.pdb')
    assert surface.frame == sasa.frames[closest_to_median(series)]
    assert surface.area == pytest.approx(series[closest_to_median(series)], rel=1e-6)
    assert surface.dots.shape[1] == 3 and len(surface.owners) == len(surface.dots)
    assert surface.atoms == gsh_aa
    # dots lie on the probe-expanded spheres of their atoms
    distance = np.linalg.norm(surface.dots - surface.positions[surface.owners], axis=1)
    assert np.allclose(distance, sasa.radii[surface.owners] + sasa.probe)
    # explicit frame (negative counts from the end), trajectory left where it was
    gsh_aa.universe.trajectory[3]
    last = sasa.write_surface(tmp_path / 'x.pdb', frame=-1)
    assert last.frame == gsh_aa.universe.trajectory.n_frames - 1
    assert gsh_aa.universe.trajectory.ts.frame == 3
    with pytest.raises(IndexError):
        sasa.write_surface(tmp_path / 'x.pdb', frame=1000)


def test_surface_reference(gsh_aa, tmp_path):
    """A reference picks the frame with the lowest RMSD, superposed onto it."""
    sasa = SASA(gsh_aa, ndots=240)
    gsh_aa.universe.trajectory[7]
    rotation = rotation_matrix(1.2, [0, 1, 1])[:3, :3]
    reference = to_nm(gsh_aa.positions) @ rotation.T + [3.0, -0.4, 0.2] # nm
    surface = sasa.write_surface(tmp_path / 'x.pdb', reference=reference)
    assert surface.frame == 7 and surface.rmsd == pytest.approx(0, abs=1e-5)
    assert np.allclose(surface.positions, reference, atol=1e-4)
    sasa.run(step=2) # then only the frames of the run: 7 is not among them
    assert sasa.write_surface(tmp_path / 'x.pdb', reference=reference).frame in sasa.frames
    with pytest.raises(ValueError, match='not both'):
        sasa.write_surface(tmp_path / 'x.pdb', frame=0, reference=reference)
    with pytest.raises(ValueError, match='reference structure has'):
        sasa.write_surface(tmp_path / 'x.pdb', reference=reference[:-1])


def test_surface_molecules(gsh_aa, tmp_path):
    """
    Per molecule, molecules do not bury each other's dots; as a whole they do;
    one molecule can be selected.
    """
    two = two_molecules(gsh_aa, 4.0) # overlapping
    single = SASA(two, ndots=240)
    single.run(stop=3)
    alone = SASA(gsh_aa, ndots=240).write_surface(tmp_path / 'x.pdb', frame=0).area
    assert single.write_surface(tmp_path / 'x.pdb', frame=0).area == pytest.approx(2 * alone, rel=1e-6)
    second = single.write_surface(tmp_path / 'x.pdb', frame=0, molecule=1)
    assert second.atoms == two[gsh_aa.n_atoms:] and second.area == pytest.approx(alone, rel=1e-6)
    with pytest.raises(IndexError):
        single.write_surface(tmp_path / 'x.pdb', frame=0, molecule=2)
    whole = SASA(two, ndots=240, per_molecule=False)
    assert whole.write_surface(tmp_path / 'x.pdb', frame=0).area < 0.9 * 2 * alone


def test_surface_pdb(gsh_aa, tmp_path):
    """Atoms as ATOM records, dots as HETATM DOT records with the resid of their atom."""
    sasa = SASA(gsh_aa, ndots=240)
    out = tmp_path / 'dots.pdb'
    surface = sasa.write_surface(out, frame=4)
    assert isinstance(surface, Surface)
    text = out.read_text()
    assert 'Frame 4' in text and 'Rowland & Taylor' in text
    with pytest.warns(UserWarning):
        written = mda.Universe(str(out))
    atoms, dots = written.select_atoms('record_type ATOM'), written.select_atoms('resname DOT')
    assert atoms.n_atoms == gsh_aa.n_atoms and list(atoms.names) == list(gsh_aa.names)
    assert dots.n_atoms == len(surface.dots) and set(dots.record_types) == {'HETATM'}
    # coordinates in nm, written in Å (PDB: 0.001 Å)
    assert np.allclose(to_nm(atoms.positions), surface.positions, atol=1e-4)
    assert np.allclose(to_nm(dots.positions), surface.dots, atol=1e-4)
    assert np.array_equal(dots.resids, gsh_aa.resids[surface.owners])


def test_cli_surface(tmp_path, capsys):
    """-surface writes the dots; molecule by label, reference structure, and errors."""
    out, dots = str(tmp_path / 'x.xvg'), tmp_path / 'dots.pdb'
    run = ['-s', TPR, '-f', XTC, '-select', 'moltype GSH', '-ndots', '240', '-skip', '50', '-o', out]
    assert main(run + ['-q', '-surface', str(dots), '-surface-molecule', 'GSH1']) == 0
    assert 'HETATM' in dots.read_text()

    # the reference: frame 100, moved; the analysed frame 100 is found again
    universe = mda.Universe(TPR, XTC)
    gsh = universe.select_atoms('moltype GSH')
    universe.trajectory[100]
    gsh.translate([10, 0, 0])
    reference = str(tmp_path / 'ref.pdb')
    gsh.write(reference)
    assert main(run + ['-surface', str(dots), '-surface-ref', reference]) == 0
    assert 'frame 100 is closest to the reference' in capsys.readouterr().err

    for options, message in [(['-whole', '-surface-molecule', '0'], 'cannot be combined with -whole'),
                             (['-surface-molecule', 'SOL3'], 'no molecule of the selection'),
                             (['-surface-molecule', '5'], 'out of range'),
                             (['-surface-frame', '9999'], 'out of range for 501 frame(s)')]:
        assert main(run + ['-q', '-surface', str(dots)] + options) == 1
        assert message in capsys.readouterr().err
    with pytest.raises(SystemExit): # frame or reference
        main(run + ['-surface', str(dots), '-surface-frame', '0', '-surface-ref', reference])
