"""
Statistics of SASA distributions by name (median, mean, mode, quartiles, ...), and the value
closest to one: to choose a representative body and frame, e.g. for a surface.
"""
from collections.abc import Callable
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde


def mode(values: ArrayLike, n_grid: int = 1024) -> float:
	"""
	Most probable value: the maximum of a Gaussian kernel density estimate of the values
	(Scott's bandwidth), on a grid of ``n_grid`` points from their minimum to maximum.
	For bimodal distributions this is the higher peak, where mean and median may fall
	between the peaks. With fewer than two distinct values, the median.
	"""
	values = np.asarray(values, dtype=float)
	if len(np.unique(values)) < 2:
		return float(np.median(values))
	grid = np.linspace(values.min(), values.max(), n_grid)
	return float(grid[np.argmax(gaussian_kde(values)(grid))])


# statistics of a SASA series by name, for choosing a representative frame
STATISTICS: dict[str, Callable[[np.ndarray], float]] = {
	'median': np.median,
	'mean': np.mean,
	'mode': mode,
	'q1': lambda values: np.percentile(values, 25),
	'q3': lambda values: np.percentile(values, 75),
	'min': np.min,
	'max': np.max,
	'mean+std': lambda values: np.mean(values) + np.std(values),
	'mean-std': lambda values: np.mean(values) - np.std(values),
}


def resolve_statistic(statistic: str | Callable[[np.ndarray], float] | float) -> tuple[Callable[[np.ndarray], float], str]:
	"""
	A statistic as a function of the values, and its name for messages: a name in
	:data:`STATISTICS`, a function of the values, or a fixed value.
	"""
	if isinstance(statistic, str):
		if statistic not in STATISTICS:
			raise ValueError(f"Unknown statistic {statistic!r}; known: {', '.join(STATISTICS)}, "
			                 f"or give a function or a value.")
		return STATISTICS[statistic], statistic
	if callable(statistic):
		name = getattr(statistic, '__name__', '')
		return statistic, name if name and name != '<lambda>' else 'custom statistic'
	value = float(statistic)
	return (lambda values: value), f"value {value:g}"


def closest_to(values: ArrayLike, statistic: str | Callable[[np.ndarray], float] | float = 'median') -> int:
	"""
	Index of the value closest to a statistic of the values, i.e. a representative frame.

	Parameters
	----------
	values : array-like
		(n,) values, e.g. the SASA of every frame
	statistic : str, callable or float
		name in :data:`STATISTICS` ('median', 'mean', 'mode', 'q1', 'q3', 'min', 'max',
		'mean+std', 'mean-std'),
		a function of the values (e.g. ``lambda v: np.percentile(v, 90)``), or a fixed value
	"""
	values = np.asarray(values)
	function, _ = resolve_statistic(statistic)
	return int(np.argmin(np.abs(values - function(values))))
