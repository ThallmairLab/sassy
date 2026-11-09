"""Plots of SASA time series."""
import numpy as np
import pytest

import MDAnalysis as mda

from fastsasa import SASA
from fastsasa.plots import (BIN_WIDTH, MAX_SERIES, Highlight, atom_sasa_at, distribution, pooled_atom_sasa,
                            save_plots)
from fastsasa.bin.fastsasa import main
from conftest import DATA

TPR, XTC = str(DATA / 'atomistic.tpr'), str(DATA / 'atomistic.xtc')


def series(n_series, n_frames=60, seed=0):
    rng = np.random.default_rng(seed)
    return np.arange(n_frames) * 10.0, rng.normal(6.0, 0.2, (n_series, n_frames))


@pytest.mark.parametrize('n_series', [1, 3, MAX_SERIES + 4])
def test_save_plots_molecules(tmp_path, n_series):
    """Molecules: total, time (more than one), distribution, panels (more than one)."""
    times, values = series(n_series)
    files = save_plots(tmp_path / 'plots', times, values)
    names = {f.name for f in files}
    expected = {'sasa_total.png', 'sasa_distribution.png'}
    if n_series > 1:
        expected |= {'sasa_time.png', 'sasa_distribution_per_molecule.png'}
    assert names == expected
    assert all(f.is_file() and f.stat().st_size > 1000 for f in files)


def test_save_plots_kinds(tmp_path):
    times, values = series(2)
    assert {f.name for f in save_plots(tmp_path, times, values, ['name S*', 'name C*'], kind='group')} == {
        'sasa_total.png', 'sasa_time.png', 'sasa_distribution.png', 'sasa_distribution_per_group.png'}
    assert {f.name for f in save_plots(tmp_path / 'w', times, values[:1], kind='whole')} == {
        'sasa_total.png', 'sasa_distribution.png'}
    # atoms: an extra panel plot, series of different lengths (pooled over molecules)
    atoms = [np.concatenate([values[0], values[1]]), values[0]]
    assert {f.name for f in save_plots(tmp_path / 'a', times, values, atoms=atoms, atom_labels=['S1', 'N1'])} == {
        'sasa_total.png', 'sasa_time.png', 'sasa_distribution.png', 'sasa_distribution_per_molecule.png',
        'sasa_distribution_per_atom.png'}
    for kind in ('atom', 'residue'):
        with pytest.raises(ValueError, match='kind'):
            save_plots(tmp_path, times, values, kind=kind)


def test_save_plots_highlight(tmp_path, monkeypatch):
    """The highlighted frame: a red line at its time and at its SASA in its series' panel."""
    from matplotlib.axes import Axes
    marks = []
    original = Axes.axvline
    monkeypatch.setattr(Axes, 'axvline', lambda ax, x, **kw: marks.append((ax, x)) or original(ax, x, **kw))
    times, values = series(3, n_frames=60) # 0 to 590 ps
    highlight = Highlight(time=200.0, sasa=float(values[1, 20]), series=1, label='surface frame 20')
    save_plots(tmp_path, times, values, ['GSH1', 'GSH2', 'GSH3'], highlight=highlight)
    titles = {ax.get_title(loc='left'): x for ax, x in marks} # titles are set after the line
    assert sum(x == 200.0 for _, x in marks) == 1 # total; the time plot has a dot instead
    assert 'SASA per molecule' not in titles
    assert titles['sasa per molecule'] == highlight.sasa # distribution
    assert titles['GSH2'] == highlight.sasa and 'GSH1' not in titles # only its panel
    # long trajectories in ns; without SASA: time plots only
    marks.clear()
    save_plots(tmp_path / 'ns', times * 100, values, highlight=Highlight(time=20000.0))
    assert [x for _, x in marks] == [20.0, 20.0]
    # the dot: on the series at the time of the frame
    files = save_plots(tmp_path / 'dot', times, values, highlight=highlight)
    assert any(f.name == 'sasa_time.png' for f in files)
    from fastsasa.plots import plot_time
    dots = []
    original_plot = Axes.plot
    monkeypatch.setattr(Axes, 'plot', lambda ax, *a, **kw: (dots.append(a) if kw.get('marker') == 'o' else None)
                        or original_plot(ax, *a, **kw))
    plot_time(times, values, tmp_path / 't.png', highlight=highlight)
    assert dots == [([200.0], [highlight.sasa])]


def test_pooled_atom_sasa(gsh_aa):
    """Every atom of a molecule type pooled over its molecules; types kept apart."""
    positions = gsh_aa.universe.trajectory.coordinate_array[:, gsh_aa.indices]
    three = mda.Merge(gsh_aa, gsh_aa, gsh_aa)
    three.add_TopologyAttr('molnums', [0, 1, 2])
    three.add_TopologyAttr('moltypes', ['GSH', 'GSH', 'GSX'])
    three.load_new(np.concatenate([positions, positions + 50.0, positions + 100.0], axis=1), order='fac')
    sasa = SASA(three.atoms, ndots=240)
    per_atom = sasa.run(step=4, per_atom=True)
    n, n_frames = gsh_aa.n_atoms, len(sasa.frames)
    labels, pooled = pooled_atom_sasa(sasa)
    assert len(labels) == len(pooled) == 2 * n # GSH (2 molecules), GSX (1)
    assert labels[0] == f"GSH {gsh_aa[0].resname}{gsh_aa[0].resid}:{gsh_aa[0].name}"
    assert labels[n].startswith('GSX ')
    assert len(pooled[0]) == 2 * n_frames and len(pooled[n]) == n_frames
    assert np.array_equal(pooled[5], np.concatenate([per_atom[5], per_atom[n + 5]]))
    assert np.array_equal(pooled[n + 5], per_atom[2 * n + 5])
    # one type: no prefix
    single = SASA(gsh_aa, ndots=240)
    single.run(stop=2)
    assert pooled_atom_sasa(single)[0][0] == f"{gsh_aa[0].resname}{gsh_aa[0].resid}:{gsh_aa[0].name}"
    # atoms without a radius (no SASA) are left out
    radii = sasa.radii.copy()
    radii[[0, n + 0, 2 * n + 3]] = 0 # first atom of both GSH, fourth of GSX
    dummy = SASA(three.atoms, ndots=240, radii=radii)
    dummy.run(stop=2)
    labels, pooled = pooled_atom_sasa(dummy)
    assert len(labels) == 2 * n - 2
    assert not any(label.endswith(f":{gsh_aa[0].name}") and label.startswith('GSH') for label in labels)
    assert f"GSX {gsh_aa[3].resname}{gsh_aa[3].resid}:{gsh_aa[3].name}" not in labels
    # the atoms of one molecule in one frame, by their panel
    frame = int(dummy.frames[1])
    for molecule, offset in ((0, 0), (1, 0), (2, n - 1)): # GSH panels first (n - 1 with a radius)
        panels = atom_sasa_at(dummy, molecule, frame)
        kept = np.flatnonzero(dummy.radii[molecule * n:(molecule + 1) * n] > 0)
        assert sorted(panels) == list(range(offset, offset + len(kept)))
        per_atom = dummy.get_sasa(dummy.molecule_atoms(molecule), per_atom=True)[kept, 1]
        assert [panels[offset + k] for k in range(len(kept))] == pytest.approx(per_atom)
        # the value lies in the pooled series of its panel
        assert all(np.any(pooled[p] == v) for p, v in panels.items())
    assert atom_sasa_at(dummy, -1, frame) == atom_sasa_at(dummy, 2, frame)
    assert atom_sasa_at(dummy, 0, 10 ** 6) == {} # not analysed


def test_distribution():
    """As fast_forward: density on 0.01 nm^2 bins on the grid of its multiples."""
    values = np.array([1.2345, 1.2399, 1.25, 1.31])
    curve = distribution(values)
    centres, density = curve.T
    assert np.allclose(np.diff(centres), BIN_WIDTH)
    assert np.allclose((centres - BIN_WIDTH / 2) / BIN_WIDTH, np.round((centres - BIN_WIDTH / 2) / BIN_WIDTH))
    # the values (bins 1.235 to 1.315) plus padding: 10 % of the range, at least 2 bins (0.02 nm^2)
    assert centres[0] == pytest.approx(1.215) and centres[-1] == pytest.approx(1.335)
    assert density[0] == density[-1] == 0
    # without padding: only the bins of the values, with the same densities
    unpadded = distribution(values, pad=0)
    assert unpadded[0, 0] == pytest.approx(1.235) and unpadded[-1, 0] == pytest.approx(1.315)
    assert np.allclose(unpadded[:, 1], density[2:2 + len(unpadded)])
    # wide distributions: 10 % of the range (0.2 nm^2 for 5 to 7 nm^2)
    wide = distribution([5.0, 7.0])
    assert wide[0, 0] == pytest.approx(4.805) and wide[-1, 0] == pytest.approx(7.205)
    assert np.sum(density) * BIN_WIDTH == pytest.approx(1.0)
    # the same as fast_forward's fixed bins (0 to 50 nm^2) where the values are
    reference, edges = np.histogram(values, density=True, bins=np.arange(0, 50, 0.01))
    centres_ff = edges[:-1] + np.diff(edges) / 2
    inside = (centres_ff > centres[0] - 1e-9) & (centres_ff < centres[-1] + 1e-9)
    assert np.allclose(reference[inside], density)
    assert distribution([]).shape == (0, 2)
    # SASA near 0: no bins below 0
    assert distribution([0.0, 0.003, 0.05])[0, 0] == pytest.approx(BIN_WIDTH / 2)


def test_cli_plots(tmp_path, capsys, monkeypatch):
    out, plots = str(tmp_path / 'x.xvg'), tmp_path / 'plots'
    run = ['-s', TPR, '-f', XTC, '-select', 'moltype GSH', '-ndots', '240', '-skip', '50', '-o', out]
    assert main(run + ['-plots', str(plots)]) == 0
    assert 'plot(s)' in capsys.readouterr().err
    assert {f.name for f in plots.iterdir()} == {'sasa_total.png', 'sasa_distribution.png'}
    # groups with a total column: the total is not plotted as a group
    assert main(run + ['-q', '-plots', str(plots / 'groups'), '-group', 'name S*', '-group', 'name O*', '-total']) == 0
    assert (plots / 'groups' / 'sasa_distribution_per_group.png').is_file()
    # per atom: the same plots as without, plus one panel per atom
    assert main(run + ['-q', '-plots', str(plots / 'atoms'), '-per-atom']) == 0
    assert {f.name for f in (plots / 'atoms').iterdir()} == {
        'sasa_total.png', 'sasa_distribution.png', 'sasa_distribution_per_atom.png'}
    assert (plots / 'atoms' / 'sasa_distribution.png').read_bytes() == (plots / 'sasa_distribution.png').read_bytes()
    # without a directory: the current one
    monkeypatch.chdir(tmp_path)
    assert main(run + ['-plots', '-q']) == 0
    assert (tmp_path / 'sasa_total.png').is_file()


def test_cli_plots_surface(tmp_path, monkeypatch):
    """The frame of the surface is highlighted in the plots."""
    import fastsasa.plots as plots
    seen = []
    original = plots.save_plots
    monkeypatch.setattr(plots, 'save_plots', lambda *a, **kw: seen.append(kw['highlight']) or original(*a, **kw))
    run = ['-s', TPR, '-f', XTC, '-select', 'moltype GSH', '-ndots', '240', '-skip', '50', '-q',
           '-o', str(tmp_path / 'x.xvg'), '-plots', str(tmp_path), '-surface', str(tmp_path / 's.pdb')]
    assert main(run) == 0
    sasa = np.loadtxt(tmp_path / 'x.xvg', comments=('#', '@'))
    highlight = seen[-1]
    row = np.flatnonzero(np.isclose(sasa[:, 0], highlight.time))[0]
    assert highlight.series == 0 and highlight.sasa == pytest.approx(sasa[row, 1], abs=1e-3)
    assert highlight.label.startswith("surface frame ") and highlight.label.endswith(" (GSH1)")
    assert highlight.atoms is None # without -per-atom
    # per atom: the atoms of the molecule in the frame, one per panel
    assert main(run + ['-per-atom']) == 0
    atoms = seen[-1].atoms
    assert sorted(atoms) == list(range(36)) and sum(atoms.values()) == pytest.approx(seen[-1].sasa)
    # a frame that was not analysed: time only
    assert main(run + ['-surface-frame', '1']) == 0
    assert seen[-1].sasa is None and seen[-1].series is None
    # whole: the selection
    assert main(run + ['-whole']) == 0
    assert seen[-1].series == 0 and seen[-1].sasa is not None
