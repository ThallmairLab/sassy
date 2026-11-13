"""Results: the SASA of every body, atom and in total, tables, and saving / loading as npz."""
import numpy as np
import pytest

from sassy import COLUMNS, SASA, Results, load
from sassy.results import save
from test_environment import copies


@pytest.fixture
def three(gsh_aa):
    """Three overlapping GSH copies (molnums 0, 1, 2; resids 1, 2, 3)."""
    return copies(gsh_aa, [0.0, 4.0, -4.0])


def test_levels(three):
    """bodies, total, by type, atoms and selections agree with each other."""
    three.residues.resnames = ['GSH', 'GSX', 'GSH']
    three.residues.moltypes = ['GSH', 'GSX', 'GSH']
    res = SASA(three.atoms, ndots=240).run(stop=4)
    assert isinstance(res, Results) and res.n_bodies == 3 and res.n_frames == 4
    assert res.body_labels == ['GSH1', 'GSX1', 'GSH2'] and res.body_type_labels == ['GSH', 'GSX', 'GSH']
    assert {name: list(indices) for name, indices in res.body_type_indices.items()} == {'GSH': [0, 2], 'GSX': [1]}
    assert np.allclose(res.total_sasa, res.body_sasa.sum(axis=0))
    assert np.allclose(sum(res.type_sasa().values()), res.total_sasa)
    assert np.array_equal(res.type_sasa(per_body=True)['GSH'], res.body_sasa[[0, 2]])
    assert np.array_equal(res.body('GSX1'), res.body_sasa[1]) and np.array_equal(res.body(-1), res.body_sasa[2])
    assert np.allclose(res.atom_sasa.sum(axis=0), res.total_sasa)
    assert np.allclose(res.select('resid 2'), res.body_sasa[1])
    assert np.array_equal(res.select([3, 1], per_atom=True), res.atom_sasa[[3, 1]]) # by index
    assert res.atom_labels[0] == 'GSH1:' + three.atoms[0].name
    assert repr(res) == '<Results: 3 bodies (GSH x2, GSX x1), 4 frames, 108 atoms>'
    with pytest.raises(KeyError, match='No body'):
        res.body('GSH3')
    with pytest.raises(IndexError):
        res.body(3)
    with pytest.raises(TypeError):
        res.select(1.5)


def test_table(three):
    """Columns of every level in the order given, then the groups, with an optional label suffix."""
    res = SASA(three.atoms, ndots=240).run(stop=3)
    labels, values = res.table()
    assert labels == res.body_labels and np.array_equal(values, res.body_sasa)
    labels, values = res.table(['total', 'types', 'bodies'], groups=['name S*', three.atoms[:2]], suffix=' x')
    assert labels == ['total x', 'GSH x', 'GSH1 x', 'GSH2 x', 'GSH3 x', 'name S* x', '2 atoms x']
    assert values.shape == (7, 3)
    assert np.allclose(values[0], res.total_sasa) and np.allclose(values[1], res.total_sasa)
    assert np.allclose(values[5], res.select('name S*'))
    labels, values = res.table(['atoms'])
    assert labels == res.atom_labels and values.shape == (three.atoms.n_atoms, 3)
    assert res.table([])[1].shape == (0, 3)
    with pytest.raises(ValueError, match='Unknown column'):
        res.table(['residues'])
    assert COLUMNS == ('bodies', 'total', 'types', 'atoms')


def test_buried_levels(three):
    """buried is free minus results at every level: bodies, total, types, atoms, selections."""
    sasa = SASA(three.select_atoms('resid 1'), ndots=240, environment='resid 2 3', buried=True)
    res = sasa.run(stop=3)
    assert res is sasa.results
    free, buried = sasa.free, sasa.buried
    assert np.allclose(buried.body_sasa, free.body_sasa - res.body_sasa)
    assert np.allclose(buried.total_sasa, free.total_sasa - res.total_sasa)
    assert np.allclose(buried.type_sasa()['GSH'], free.type_sasa()['GSH'] - res.type_sasa()['GSH'])
    assert np.allclose(buried.atom_sasa, free.atom_sasa - res.atom_sasa)
    assert np.allclose(buried.select('name O*'), free.select('name O*') - res.select('name O*'))
    assert buried.body_labels == res.body_labels and np.array_equal(buried.times, res.times)


def test_save_load(three, tmp_path):
    """Everything is saved, and loaded back the same; selections by string need the universe."""
    sasa = SASA(three.select_atoms('resid 1 2'), ndots=240, environment='resid 3', buried=True)
    sasa.run(stop=3)
    out = tmp_path / 'sasa.npz'
    sasa.save(out, table=sasa.results.table(['bodies', 'total']), selection='resid 1 2')
    saved = load(out)
    for name in ('results', 'free', 'buried'):
        a, b = getattr(saved, name), getattr(sasa, name)
        assert np.allclose(a.body_sasa, b.body_sasa) and np.allclose(a.atom_sasa, b.atom_sasa)
        assert a.body_labels == b.body_labels and a.body_type_labels == b.body_type_labels and a.atom_labels == b.atom_labels
        assert np.array_equal(a.frames, b.frames) and np.allclose(a.times, b.times)
    assert saved.meta['selection'] == 'resid 1 2' and saved.meta['bodies'] == 'molecule'
    assert saved.meta['ndots'] == len(sasa.dots) and saved.meta['n_environment'] == 36
    # by index without the universe, by selection string with it
    assert np.allclose(saved.results.select([0, 1]), sasa.results.select([0, 1]))
    with pytest.raises(ValueError, match='universe'):
        saved.results.select('name S*')
    with_universe = load(out, three)
    assert np.allclose(with_universe.buried.select('name S*'), sasa.buried.select('name S*'))
    # plain numpy: the table and the arrays
    data = np.load(out)
    assert data['column_labels'].tolist() == ['GSH1', 'GSH2', 'total']
    assert np.allclose(data['free_total_sasa'], sasa.free.total_sasa)
    # without atoms and without free: only the bodies
    light = SASA(three.atoms, ndots=240, keep_atoms=False)
    save(tmp_path / 'light.npz', light.run(stop=2), meta={'note': 'bodies only'})
    saved = load(tmp_path / 'light.npz')
    assert saved.results.atom_sasa is None and saved.free is None and saved.buried is None
    assert saved.meta == {'note': 'bodies only'}
    assert np.allclose(saved.results.total_sasa, light.results.total_sasa)
