"""Results: the SASA of every body, atom and in total, tables, and saving / loading as npz."""
import numpy as np
import pytest

from sassy import COLUMNS, SASA, Results, load
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
    assert res.body_labels == ['GSH1', 'GSX1', 'GSH2'] and res.body_types == ['GSH', 'GSX', 'GSH']
    assert {name: list(indices) for name, indices in res.types.items()} == {'GSH': [0, 2], 'GSX': [1]}
    assert np.allclose(res.total, res.bodies.sum(axis=0))
    assert np.allclose(sum(res.by_type().values()), res.total)
    assert np.array_equal(res.by_type(per_body=True)['GSH'], res.bodies[[0, 2]])
    assert np.array_equal(res.body('GSX1'), res.bodies[1]) and np.array_equal(res.body(-1), res.bodies[2])
    assert np.allclose(res.atoms.sum(axis=0), res.total)
    assert np.allclose(res.select('resid 2'), res.bodies[1])
    assert np.array_equal(res.select([3, 1], per_atom=True), res.atoms[[3, 1]]) # by index
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
    assert labels == res.body_labels and np.array_equal(values, res.bodies)
    labels, values = res.table(['total', 'types', 'bodies'], groups=['name S*', three.atoms[:2]], suffix=' x')
    assert labels == ['total x', 'GSH x', 'GSH1 x', 'GSH2 x', 'GSH3 x', 'name S* x', '2 atoms x']
    assert values.shape == (7, 3)
    assert np.allclose(values[0], res.total) and np.allclose(values[1], res.total)
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
    assert np.allclose(buried.bodies, free.bodies - res.bodies)
    assert np.allclose(buried.total, free.total - res.total)
    assert np.allclose(buried.by_type()['GSH'], free.by_type()['GSH'] - res.by_type()['GSH'])
    assert np.allclose(buried.atoms, free.atoms - res.atoms)
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
        assert np.allclose(a.bodies, b.bodies) and np.allclose(a.atoms, b.atoms)
        assert a.body_labels == b.body_labels and a.body_types == b.body_types and a.atom_labels == b.atom_labels
        assert np.array_equal(a.frames, b.frames) and np.allclose(a.times, b.times)
    assert saved.meta['selection'] == 'resid 1 2' and saved.meta['bodies'] == 'molecule'
    assert saved.meta['ndots'] == len(sasa.unit) and saved.meta['n_environment'] == 36
    # by index without the universe, by selection string with it
    assert np.allclose(saved.results.select([0, 1]), sasa.results.select([0, 1]))
    with pytest.raises(ValueError, match='universe'):
        saved.results.select('name S*')
    with_universe = load(out, three)
    assert np.allclose(with_universe.buried.select('name S*'), sasa.buried.select('name S*'))
    # plain numpy: the table and the arrays
    data = np.load(out)
    assert data['column_labels'].tolist() == ['GSH1', 'GSH2', 'total']
    assert np.allclose(data['free_total'], sasa.free.total)
    # without atoms and without free: only the bodies
    light = SASA(three.atoms, ndots=240, keep_atoms=False)
    light.run(stop=2).save(tmp_path / 'light.npz', note='bodies only')
    saved = load(tmp_path / 'light.npz')
    assert saved.results.atoms is None and saved.free is None and saved.buried is None
    assert saved.meta == {'note': 'bodies only'}
    assert np.allclose(saved.results.total, light.results.total)
