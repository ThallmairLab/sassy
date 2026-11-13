"""Molecules from bonds, molecule types and labels, and plots by type."""
import logging
import numpy as np
import pytest
import MDAnalysis as mda

from sassy import SASA, molecule_index, numbered_labels, molecule_types
from sassy.plots import Highlight, save_plots


def mixed(gsh, types=('GSH', 'GSX', 'GSH'), molnums=True, bonds=True):
    """Copies of GSH, apart from each other; GSX: a renamed copy (another type)."""
    positions = gsh.universe.trajectory.coordinate_array[:, gsh.indices]
    u = mda.Merge(*[gsh] * len(types))
    u.load_new(np.concatenate([positions + 50.0 * i for i in range(len(types))], axis=1), order='fac')
    u.add_TopologyAttr('moltypes', list(types))
    u.residues.resnames = list(types)
    if molnums:
        u.add_TopologyAttr('molnums', np.arange(len(types)))
    else:
        u.del_TopologyAttr('molnums')
        u.del_TopologyAttr('moltypes')
    if not bonds:
        u.del_TopologyAttr('bonds')
    return u


def test_molecules_from_bonds(gsh_aa):
    """Without molnums, bonds split any mix of molecules; without both, a mix raises an error."""
    n = gsh_aa.n_atoms
    u = mixed(gsh_aa, molnums=False)
    assert molecule_index(u.atoms).tolist() == np.repeat([0, 1, 2], n).tolist()
    # a selection cutting molecules: the selected part of every molecule
    assert molecule_index(u.atoms[n - 2:n + 2]).tolist() == [0, 0, 1, 1]
    # neither molnums nor bonds: a mix does not repeat
    u = mixed(gsh_aa, molnums=False, bonds=False)
    with pytest.raises(ValueError, match='No molecules found: the topology has neither molecules'):
        molecule_index(u.atoms)
    # copies of one type still repeat
    assert molecule_index(mixed(gsh_aa, ('GSH', 'GSH'), molnums=False, bonds=False).atoms).max() == 1


def test_molecule_types(gsh_aa):
    u = mixed(gsh_aa) # moltypes
    assert molecule_types(u.atoms) == ['GSH', 'GSX', 'GSH']
    assert numbered_labels(molecule_types(u.atoms)) == ['GSH1', 'GSX1', 'GSH2']
    u = mixed(gsh_aa, molnums=False) # by atom and residue names
    assert molecule_types(u.atoms) == ['GSH', 'GSX', 'GSH']
    # same first residue name, different atoms: told apart
    u.atoms[gsh_aa.n_atoms + 3].name = 'XX'
    u.residues[1].resname = 'GSH'
    assert molecule_types(u.atoms) == ['GSH', 'GSH_2', 'GSH'] and len(set(molecule_types(u.atoms))) == 2
    assert molecule_types(u.atoms[:0]) == [] and len(set(molecule_types(u.atoms[:0]))) == 0


def test_sasa_types(gsh_aa):
    sasa = SASA(mixed(gsh_aa).atoms, ndots=240)
    assert sasa.body_labels == ['GSH1', 'GSX1', 'GSH2']
    assert {name: list(indices) for name, indices in sasa.body_type_indices.items()} == {'GSH': [0, 2], 'GSX': [1]}
    sasa.run(stop=3)
    assert np.array_equal(sasa.results.body('GSH2'), sasa.results.body(2))
    assert np.array_equal(sasa.results.type_sasa(per_body=True)['GSH'], sasa.results.body_sasa[[0, 2]])
    assert np.allclose(sasa.results.type_sasa()['GSH'] + sasa.results.type_sasa()['GSX'],
                       sasa.results.total_sasa)
    assert sasa.body_atoms('GSX1') == sasa.body_atoms(1)
    with pytest.raises(KeyError, match='GSH3'):
        sasa.results.body('GSH3')
    with pytest.raises(KeyError, match='No body type'):
        sasa.results.representative('SOL')
    assert list(sasa.results.type_sasa()) == ['GSH', 'GSX'] and sasa.results.body_type_indices.keys() == sasa.body_type_indices.keys()
    whole = SASA(mixed(gsh_aa).atoms, ndots=240, bodies='selection')
    assert whole.body_labels == ['selection'] and list(whole.body_type_indices) == ['selection']


def test_plots_by_type(tmp_path, monkeypatch):
    """Several types: one distribution per type, one mean per type, panels ordered and colored by type."""
    from matplotlib.axes import Axes
    lines = []
    original = Axes.plot
    monkeypatch.setattr(Axes, 'plot', lambda ax, *a, **kw: lines.append(kw) or original(ax, *a, **kw))
    rng = np.random.default_rng(0)
    times, values = np.arange(50) * 10.0, rng.normal(6.0, 0.2, (12, 50))
    types = ['GSH', 'SOL'] * 6
    labels = numbered_labels(types)
    highlight = Highlight(time=100.0, sasa=float(values[3, 10]), series=3, label='surface frame 10 (SOL2)')
    marks = {}
    from sassy import plots
    panels = plots.plot_distribution_panels
    monkeypatch.setattr(plots, 'plot_distribution_panels',
                        lambda v, f, l, **kw: marks.update(labels=l, marks=kw['marks'], colors=kw['colors'])
                        or panels(v, f, l, **kw))
    calls = []
    monkeypatch.setattr(plots, 'plot_distribution_panels',
                        lambda v, f, l, **kw: calls.append((f.name, v, l, kw))
                        or marks.update(labels=l, marks=kw['marks'], colors=kw['colors']) or panels(v, f, l, **kw))
    save_plots(tmp_path, times, values, labels, highlight=highlight, types=types)
    named = [kw.get('label') for kw in lines if kw.get('label')]
    # distribution: one panel per type, pooled over its molecules
    name, pooled, type_labels, kw = calls[0]
    assert name == 'sasa_distribution.png' and type_labels == ['GSH (6 molecules)', 'SOL (6 molecules)']
    assert np.array_equal(pooled[1], values[1::2].ravel())
    assert kw['marks'] == {1: [(highlight.sasa, highlight.label)]} and kw['colors'] == plots.SERIES_COLORS[:2]
    assert marks['labels'] == [f"GSH{i}" for i in range(1, 7)] + [f"SOL{i}" for i in range(1, 7)]
    assert marks['marks'] == {7: [(highlight.sasa, highlight.label)]} # SOL2 is the 8th panel
    assert marks['colors'] == [plots.SERIES_COLORS[0]] * 6 + [plots.SERIES_COLORS[1]] * 6
    # one type: as before (no per-type curves, one color)
    lines.clear()
    save_plots(tmp_path / 'one', times, values, labels, types=['GSH'] * 12)
    assert not any(': mean of' in (kw.get('label') or '') or ' molecules)' in (kw.get('label') or '') for kw in lines)
    assert marks['colors'] is None
    with pytest.raises(ValueError, match='types'):
        save_plots(tmp_path, times, values, labels, types=['GSH'])


def test_time_panels_by_type(tmp_path, monkeypatch):
    """sasa_time: one panel per type stacked vertically; the dot in the panel of its molecule."""
    from sassy import plots
    figures = []
    save = plots._save
    monkeypatch.setattr(plots, '_save', lambda fig, fname: figures.append(fig) or save(fig, fname))
    rng = np.random.default_rng(1)
    times, values = np.arange(30) * 10.0, rng.normal(6.0, 0.2, (13, 30))
    types = ['GSH'] * 3 + ['SOL'] * 10 # SOL: more than 8, faint lines and a mean
    labels = numbered_labels(types)
    highlight = Highlight(time=50.0, sasa=float(values[4, 5]), series=4, label='surface frame 5 (SOL2)')
    plots.plot_time(times, values, tmp_path / 't.png', labels, highlight=highlight, types=types)
    axes = figures[-1].axes
    assert [ax.get_title(loc='left') for ax in axes] == ['GSH (3 molecules)', 'SOL (10 molecules)']
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[1]) # one time axis
    assert axes[0].get_position().y0 > axes[1].get_position().y0 # stacked vertically

    def dots(ax):
        return [(list(line.get_xdata()), list(line.get_ydata())) for line in ax.lines if line.get_marker() == 'o']
    assert dots(axes[0]) == [] and dots(axes[1]) == [([50.0], [highlight.sasa])]
    assert any(line.get_label() == 'mean of 10 molecules' for line in axes[1].lines)
    assert any(line.get_label() == 'GSH2' for line in axes[0].lines)
    # SASA not known: a vertical line in every panel
    plots.plot_time(times, values, tmp_path / 't.png', labels, highlight=Highlight(time=50.0), types=types)
    assert all(any(line.get_linestyle() == '--' and list(line.get_xdata()) == [50.0, 50.0] for line in ax.lines)
               for ax in figures[-1].axes)
    # one type: one panel as before
    plots.plot_time(times, values, tmp_path / 't.png', labels)
    assert len(figures[-1].axes) == 1


def test_surface_per_type(gsh_aa, tmp_path):
    """Every type its own statistic (over all its molecules and frames) and representative."""
    u = mixed(gsh_aa, ('GSH', 'GSX', 'GSH'))
    # GSX smaller: no atom radius for its hydrogens -> another distribution
    radii = SASA(u.atoms).radii.copy()
    n = gsh_aa.n_atoms
    radii[n:2 * n][np.char.startswith(u.atoms[n:2 * n].names.astype(str), 'H')] = 0
    sasa = SASA(u.atoms, ndots=240, radii=radii)
    sasa.run(step=2)
    for name in ('GSH', 'GSX'):
        values = sasa.results.type_sasa(per_body=True)[name]
        for statistic in ('median', 'mean', 'q3'):
            molecule, frame, target = sasa.results.representative(name, statistic)
            from sassy.statistics import STATISTICS
            assert target == pytest.approx(STATISTICS[statistic](values.ravel()))
            column = np.flatnonzero(sasa.results.frames == frame)[0]
            assert molecule in sasa.body_type_indices[name]
            assert abs(sasa.results.body_sasa[molecule, column] - target) == pytest.approx(np.abs(values - target).min())
    assert sasa.results.representative('GSH')[2] != pytest.approx(sasa.results.representative('GSX')[2]) # own medians
    # a fixed frame: only the molecule is chosen
    frame = int(sasa.results.frames[3])
    molecule, chosen, target = sasa.results.representative('GSH', frame=frame)
    assert chosen == frame and molecule == [0, 2][np.argmin(np.abs(sasa.results.body_sasa[[0, 2], 3] - target))]
    assert sasa.results.representative('GSH', frame=1)[:2] == (0, 1) # not analysed: the first molecule
    # write_surface by type
    surface = sasa.surface(body_type='GSX')
    assert surface.body == 1 and surface.frame == sasa.results.representative('GSX')[1]
    assert any(r.startswith('Representative molecule and frame: GSX1 in frame') and
               '1 GSX molecule(s) x' in r for r in surface.remarks)
    # one molecule: its frame closest to the statistic of its type
    target = np.median(sasa.results.type_sasa(per_body=True)['GSH'])
    surface = sasa.surface(body='GSH2')
    assert surface.frame == sasa.results.frames[np.argmin(np.abs(sasa.results.body(2) - target))]
    with pytest.raises(KeyError):
        sasa.surface(body_type='SOL').write(tmp_path / 'x.pdb')


def test_plots_several_highlights(tmp_path, monkeypatch):
    """One highlight per type: each in the panels of its type, labels once in the legends."""
    from sassy import plots
    calls = []
    panels = plots.plot_distribution_panels
    monkeypatch.setattr(plots, 'plot_distribution_panels',
                        lambda v, f, l, **kw: calls.append((f.name, kw['marks'])) or panels(v, f, l, **kw))
    rng = np.random.default_rng(2)
    times, values = np.arange(30) * 10.0, rng.normal(6.0, 0.2, (4, 30))
    types = ['GSH', 'SOL', 'GSH', 'SOL']
    h = [Highlight(50.0, float(values[2, 5]), 2, 'surface frame 5 (GSH2)', atoms={0: 0.1}),
         Highlight(70.0, float(values[1, 7]), 1, 'surface frame 7 (SOL1)', atoms={3: 0.2})]
    save_plots(tmp_path, times, values, numbered_labels(types), highlight=h, types=types,
               atoms=[rng.random(10) for _ in range(4)], atom_labels=list('abcd'))
    marks = dict(calls)
    assert marks['sasa_distribution.png'] == {0: [(h[0].sasa, h[0].label)], 1: [(h[1].sasa, h[1].label)]}
    # panels ordered by type: GSH1, GSH2, SOL1, SOL2
    assert marks['sasa_distribution_per_molecule.png'] == {1: [(h[0].sasa, h[0].label)], 2: [(h[1].sasa, h[1].label)]}
    assert marks['sasa_distribution_per_atom.png'] == {0: [(0.1, h[0].label)], 3: [(0.2, h[1].label)]}


def test_residue_bodies(gsh_aa, tmp_path):
    """
    Every residue a body: a single-residue molecule as per molecule; the residues of one
    molecule on their own (they do not bury each other), labelled <resname><resid>.
    """
    from sassy import residue_labels, residue_types
    from test_environment import copies
    single = SASA(gsh_aa, ndots=240, bodies='residue')
    assert single.body_labels == ['GSH1'] and single.body_type_labels == ['GSH']
    assert np.allclose(single.run(stop=4).body_sasa, SASA(gsh_aa, ndots=240).run(stop=4).body_sasa, rtol=1e-12)

    u = copies(gsh_aa, [0.0, 4.0, -4.0]) # three overlapping residues ...
    u.residues.molnums = [0, 0, 0] # ... of one molecule
    molecule = SASA(u.atoms, ndots=240)
    assert molecule.n_bodies == 1
    molecule.run(stop=4)
    residues = SASA(u.atoms, ndots=240, bodies='residue')
    assert residues.body_labels == ['GSH1', 'GSH2', 'GSH3'] and list(residues.body_type_indices) == ['GSH']
    residues.run(stop=4)
    alone = SASA(gsh_aa, ndots=240).run(stop=4).body_sasa[0]
    assert np.allclose(residues.results.body_sasa, alone, rtol=1e-6) # each residue on its own
    for k in range(3): # in the molecule, a residue is buried by the others
        assert np.all(molecule.results.select(f'resid {k + 1}') < residues.results.body_sasa[k])
    assert 'Residue 1 of 3 (GSH, resid 2, 36 atoms), on its own' in \
        residues.surface(body='GSH2').remarks

    # duplicate resids: the segid, else a number; other atoms: another type
    u.residues.resids = [1, 1, 1]
    assert residue_labels(u.atoms) == ['GSH1', 'GSH1_2', 'GSH1_3']
    u.segments.segids = ['A', 'B', 'C']
    assert residue_labels(u.atoms) == ['GSH1:A', 'GSH1:B', 'GSH1:C']
    u.atoms[2 * gsh_aa.n_atoms + 3].name = 'XX'
    assert residue_types(u.atoms) == ['GSH', 'GSH', 'GSH_2']


def test_bodies_argument(gsh_aa):
    """The removed per_molecule keyword and unknown bodies are rejected."""
    assert SASA(gsh_aa, ndots=240, bodies='selection').body_labels == ['selection']
    with pytest.raises(TypeError, match='per_molecule'):
        SASA(gsh_aa, per_molecule=False)
    with pytest.raises(ValueError, match='bodies must be one of'):
        SASA(gsh_aa, bodies='chain')
