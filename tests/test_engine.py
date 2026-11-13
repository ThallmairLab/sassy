"""Burial test and surface dots."""
import numpy as np
import pytest

import logging

from sassy import unit_dots, surface_dots
from sassy.engine import _exposed_counts, _exposed_dots


def test_single_sphere():
    """
    An isolated sphere keeps all its dots on the probe-expanded sphere.
    """
    dots, owners, area = surface_dots(np.zeros((1, 3)), np.array([0.2]), probe=0.1, ndots=240)
    assert len(dots) == 252
    assert np.all(owners == 0)
    assert np.allclose(np.linalg.norm(dots, axis=1), 0.3)
    assert area == pytest.approx(4 * np.pi * 0.3**2)


def test_two_spheres_area():
    """
    Two overlapping spheres: the dot area approaches the analytic value and
    no dot lies inside the other sphere.
    """
    radius, probe, dist = 0.2, 0.1, 0.4
    xyz = np.array([[0.0, 0.0, 0.0], [dist, 0.0, 0.0]])
    dots, owners, area = surface_dots(xyz, np.array([radius, radius]), probe=probe, ndots=4000)
    ext = radius + probe
    cap = 2 * np.pi * ext * (ext - dist / 2)
    assert area == pytest.approx(2 * (4 * np.pi * ext**2 - cap), rel=0.01)
    for i in range(2):
        other = xyz[1 - i]
        assert np.all(np.linalg.norm(dots[owners == i] - other, axis=1) >= ext - 1e-9)


def test_zero_radius_ignored():
    """
    Atoms without radius (dummies, virtual sites) neither carry dots nor bury others.
    """
    xyz = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    dots, owners, area = surface_dots(xyz, np.array([0.2, 0.0]), probe=0.1, ndots=240)
    assert len(dots) == 252
    assert np.all(owners == 0)
    assert area == pytest.approx(4 * np.pi * 0.3**2)


def _brute_force_areas(xyz, radii, probe, unit):
    """
    Per-sphere exposed area, testing every dot against every other sphere.
    """
    ext = radii + probe
    areas = []
    for i in range(len(xyz)):
        dots = xyz[i] + ext[i] * unit
        dist = np.linalg.norm(dots[:, None] - xyz[None], axis=2)
        inside = dist < ext[None]
        inside[:, i] = False
        areas.append(np.mean(~inside.any(axis=1)) * 4 * np.pi * ext[i]**2)
    return np.array(areas)


def test_surface_dots_brute_force():
    """
    The neighbour lists and the compiled burial test agree with testing every
    dot against every sphere, for a dense random cluster with mixed radii.
    """
    rng = np.random.default_rng(3)
    xyz = rng.uniform(0, 1.2, size=(60, 3))
    radii = rng.choice([0.109, 0.156, 0.175, 0.264], size=60)
    unit = unit_dots(240)
    _, owners, area = surface_dots(xyz, radii, probe=0.191, ndots=240)
    expected = _brute_force_areas(xyz, radii, 0.191, unit)
    assert area == pytest.approx(expected.sum(), rel=1e-12)
    counts = np.bincount(owners, minlength=60)
    assert np.allclose(counts / len(unit) * 4 * np.pi * (radii + 0.191)**2, expected)


def test_counts_equal_mask():
    """The count-only kernel gives the number of exposed dots of the mask."""
    rng = np.random.default_rng(5)
    xyz = rng.uniform(0, 2.0, size=(300, 3))
    ext = rng.choice([0.3, 0.35, 0.45], size=300)
    block = rng.integers(0, 3, size=300)
    unit = unit_dots(500)
    mask, pairs_mask = _exposed_dots(xyz, ext, block, unit)
    counts, pairs_counts = _exposed_counts(xyz, ext, block, unit)
    assert np.array_equal(counts, mask.sum(axis=1))
    assert len(pairs_mask) == len(pairs_counts) == 0


def test_coincident_warning(caplog):
    """
    Atoms with a radius at an identical position are warned about (with equal radii
    neither buries the other); not if one has radius 0 or they are in different blocks.
    """
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    with caplog.at_level(logging.WARNING, logger='sassy'):
        surface_dots(xyz, [0.2, 0.2, 0.2], ndots=240)
    assert '1 pair(s) of atoms with a non-zero radius share an identical position' in caplog.text
    assert 'atom 0/atom 1' in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='sassy'):
        surface_dots(xyz, [0.2, 0.0, 0.2], ndots=240)
        surface_dots(xyz, [0.2, 0.2, 0.2], ndots=240, block=[0, 1, 0])
    assert 'identical position' not in caplog.text


def test_target_mask():
    """
    Spheres that are no target bury the targets of their block as before, but are not
    buried themselves (no neighbour list: all their dots are reported exposed).
    """
    rng = np.random.default_rng(7)
    xyz = rng.uniform(0, 2.0, size=(300, 3))
    ext = rng.choice([0.3, 0.35, 0.45], size=300)
    block = rng.integers(0, 3, size=300)
    target = rng.random(300) < 0.4
    unit = unit_dots(500)
    full, _ = _exposed_counts(xyz, ext, block, unit)
    masked, _ = _exposed_counts(xyz, ext, block, unit, target)
    assert np.array_equal(masked[target], full[target])
    assert np.all(masked[~target] == len(unit))
    mask, _ = _exposed_dots(xyz, ext, block, unit, target)
    assert np.array_equal(mask.sum(axis=1), masked)
    with pytest.raises(ValueError, match='one value per sphere'):
        _exposed_counts(xyz, ext, block, unit, target[:-1])


def test_target_surface_dots(caplog):
    """surface_dots with a target: dots and area of the targets only, buried by the others."""
    xyz = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    both_dots, both_owners, both_area = surface_dots(xyz, [0.2, 0.2], probe=0.1, ndots=1000)
    dots, owners, area = surface_dots(xyz, [0.2, 0.2], probe=0.1, ndots=1000, target=[True, False])
    assert np.all(owners == 0) and len(dots) == np.sum(both_owners == 0)
    assert area == pytest.approx(both_area / 2) # symmetric pair: half of the total
    # coincident with an atom that is no target: still warned about
    with caplog.at_level(logging.WARNING, logger='sassy'):
        surface_dots(np.zeros((2, 3)), [0.2, 0.2], ndots=240, target=[True, False])
    assert 'identical position' in caplog.text
