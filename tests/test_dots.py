"""Dots on the unit sphere from geodesic subdivisions."""
import numpy as np
import pytest

from sassy import unit_dots


@pytest.mark.parametrize('ndots, expected', [(24, 32), (50, 92), (100, 122), (240, 252)])
def test_unit_dots(ndots, expected):
    """
    The number of dots is the smallest geodesic count >= ndots and all dots lie on the unit sphere.
    """
    dots = unit_dots(ndots)
    assert len(dots) == expected
    assert np.allclose(np.linalg.norm(dots, axis=1), 1.0)
    # no duplicated dots
    assert len(np.unique(np.round(dots, 6), axis=0)) == expected
