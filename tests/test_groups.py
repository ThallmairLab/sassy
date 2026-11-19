"""Groups of atoms: weights of shared atoms, SASA of groups during the run and from the atoms, body members."""
import MDAnalysis as mda
import numpy as np
import pytest

from sassy import SASA, group_weights, load, radii_references, radii_remarks
from sassy.radii import BONDI, ROWLAND_TAYLOR
from test_environment import copies


@pytest.fixture
def three(gsh_aa):
    """Three overlapping GSH copies (molnums 0, 1, 2; resids 1, 2, 3)."""
    return copies(gsh_aa, [0.0, 4.0, -4.0])


# group_weights
def test_group_weights_equal_share():
    """A shared atom gives 1/k to each of its k groups; duplicates count once; unassigned atoms are listed."""
    weights, unassigned = group_weights([[0, 1, 1], [1, 2], [2, 1], []], 5)
    assert weights.shape == (4, 5)
    assert np.allclose(weights.toarray(), [[1, 1 / 3, 0, 0, 0],
                                           [0, 1 / 3, 0.5, 0, 0],
                                           [0, 1 / 3, 0.5, 0, 0],
                                           [0, 0, 0, 0, 0]])
    assert unassigned.tolist() == [3, 4]
    # every assigned atom is split completely
    assert np.allclose(weights.sum(axis=0).A1, [1, 1, 1, 0, 0])


def test_group_weights_full_share():
    weights, _ = group_weights([[0, 1], [1, 2]], 3, share='full')
    assert np.array_equal(weights.toarray(), [[1, 1, 0], [0, 1, 1]])


def test_group_weights_no_groups():
    weights, unassigned = group_weights([], 3)
    assert weights.shape == (0, 3) and unassigned.tolist() == [0, 1, 2]


@pytest.mark.parametrize('groups, share, error', [
    ([[0, 3]], 'equal', IndexError),
    ([[-1]], 'equal', IndexError),
    ([[0.5]], 'equal', TypeError),
    ([[0]], 'half', ValueError),
])
def test_group_weights_invalid(groups, share, error):
    with pytest.raises(error):
        group_weights(groups, 3, share=share)


# SASA of groups
def test_groups_during_run_match_atoms(three):
    """
    Groups computed during the run without the per-atom SASA equal those from the per-atom SASA,
    and with equal sharing the groups and the atoms in none add up to the total.
    """
    groups = {'head': 'resid 1 and index 0:9', 'middle': 'resid 1 and index 5:20', 'other': 'resid 2'}
    streamed = SASA(three.atoms, ndots=240, groups=groups, keep_atoms=False)
    res = streamed.run(stop=4)
    assert res.atom_sasa is None and res.group_labels == ['head', 'middle', 'other']
    assert res.group_sasa.shape == (3, 4)
    assert repr(res).endswith(', no atoms kept, 3 groups>')

    full = SASA(three.atoms, ndots=240).run(stop=4)
    assert full.group_sasa is None and full.group_labels is None
    assert np.allclose(full.group(list(groups.values())), res.group_sasa)

    # equal share: groups + atoms in no group = total
    ungrouped = full.select(streamed.ungrouped)
    assert np.allclose(res.group_sasa.sum(axis=0) + ungrouped, full.total_sasa)
    assert np.allclose(res.group_sasa[2], full.body_sasa[1]) # a whole molecule

    # full share: the shared atoms count for every group
    shared = full.select('resid 1 and index 5:9')
    full_share = full.group(list(groups.values()), share='full')
    assert np.allclose(full_share.sum(axis=0), res.group_sasa.sum(axis=0) + shared)


def test_group_labels_and_kinds(three):
    """Groups as selection strings, AtomGroups or positions; labelled by dict key, string or number."""
    ag = three.atoms
    sasa = SASA(ag, ndots=120, groups=['resid 1', ag.select_atoms('resid 2'), np.arange(72, 108)])
    res = sasa.run(stop=2)
    assert res.group_labels == ['resid 1', 'group2', 'group3']
    assert np.allclose(res.group_sasa, res.body_sasa)
    with pytest.raises(TypeError, match='not a single group'):
        SASA(ag, groups='resid 1')
    with pytest.raises(ValueError, match='not in the AtomGroup'):
        SASA(three.select_atoms('resid 1'), groups=[ag.select_atoms('resid 2')])


def test_set_groups(three):
    """Groups set after construction, e.g. from the bodies, are computed by the next run; None removes them."""
    sasa = SASA(three.atoms, ndots=120, keep_atoms=False)
    sasa.set_groups([members[:10] for members in sasa.body_members], share='full')
    res = sasa.run(stop=2)
    assert res.group_labels == ['group1', 'group2', 'group3'] and res.group_sasa.shape == (3, 2)
    assert len(sasa.ungrouped) == 108 - 30
    sasa.set_groups(None)
    assert sasa.run(stop=2).group_sasa is None and sasa.ungrouped is None


def test_group_needs_atoms(three):
    res = SASA(three.atoms, ndots=120, keep_atoms=False).run(stop=1)
    with pytest.raises(RuntimeError, match='groups=...'):
        res.group(['resid 1'])


def test_groups_save_load_buried(three, tmp_path):
    """Groups are saved and loaded, also without the environment and buried by it."""
    sasa = SASA(three.select_atoms('resid 1'), ndots=240, environment='resid 2 3', buried=True,
                groups={'a': 'index 0:9', 'b': 'index 10:35'})
    res = sasa.run(stop=3)
    assert np.allclose(sasa.buried.group_sasa, sasa.free.group_sasa - res.group_sasa)
    assert sasa.buried.group_labels == ['a', 'b']
    sasa.save(tmp_path / 'groups.npz')
    saved = load(tmp_path / 'groups.npz')
    assert saved.results.group_labels == ['a', 'b']
    assert np.allclose(saved.results.group_sasa, res.group_sasa)
    assert np.allclose(saved.free.group_sasa, sasa.free.group_sasa)
    assert np.allclose(saved.buried.group_sasa, sasa.buried.group_sasa)


# bodies
def test_body_members(three):
    sasa = SASA(three.atoms, ndots=120)
    res = sasa.run(stop=1)
    expected = [np.arange(36 * k, 36 * (k + 1)) for k in range(3)]
    assert [m.tolist() for m in sasa.body_members] == [m.tolist() for m in expected]
    assert [m.tolist() for m in res.body_members] == [m.tolist() for m in expected]
    sasa.body_members[0][0] = -1 # a copy
    assert sasa.body_members[0][0] == 0


# radii references
def test_radii_references(gsh_aa):
    assert radii_references(gsh_aa) == [ROWLAND_TAYLOR]
    assert radii_references(gsh_aa, cg=True) == []
    universe = mda.Universe.empty(2, trajectory=True)
    universe.add_TopologyAttr('names', ['P1', 'C1'])
    universe.add_TopologyAttr('elements', ['P', 'C'])
    assert [ref.key for ref in radii_references(universe.atoms)] == ['rowland_taylor_1996', 'bondi_1964']
    assert BONDI.doi == '10.1021/j100785a001'
    assert radii_remarks(universe.atoms)[:2] == ["Atomistic vdW radii: Rowland & Taylor (1996), doi:10.1021/jp953141+",
                                                 "Phosphorus radius: Bondi (1964), doi:10.1021/j100785a001"]
