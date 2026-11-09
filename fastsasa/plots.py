"""
Simple plots of SASA time series: time evolution, convergence (running mean) and
distributions, saved as PNG files. Distributions are computed and drawn as in
fast_forward (0.01 nm^2 bins, density lines zoomed to the populated range).

The functions take plain arrays, so they work for molecules, groups or atoms alike:
``values`` is (n_series, n_frames) SASA in nm^2 and ``times`` (n_frames,) in ps.

	>>> sasa.run()
	>>> save_plots('plots', sasa.times, sasa.molecule_sasa, total=sasa.total_sasa)
"""
import logging
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# style (static PNG, light surface): categorical hues in fixed order, never cycled;
# text in neutral ink, never in series colors; recessive grid and axes
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'
MUTED = '#a8a7a1' # individual series behind a summary line
MAX_SERIES = len(SERIES_COLORS) # more series are summarised, not given generated colors
MAX_PANELS = 25 # small multiples
MAX_ATOM_PANELS = 100 # small multiples of atoms
LINE_WIDTH = 1.5 # pt, ~2 px at 100 dpi
DPI = 150

# distributions as in fast_forward (interaction_plots.make_distribution_plot)
BIN_WIDTH = 0.01 # nm^2
DISTRIBUTION_COLOR = '#6970E0'
ZOOM_THRESHOLD = 0.01 # x range where the density exceeds 1 % of its maximum ...
PAD_FRACTION = 0.1 # ... plus this fraction of that range on both sides (fast_forward: 0.25 nm^2),
MIN_PAD_BINS = 2 # ... but at least this many bins
SASA_LABEL = 'SASA [nm²]'
DENSITY_LABEL = 'Probability Density'


def _figure(width: float = 7.0, height: float = 4.0, nrows: int = 1, ncols: int = 1, **kwargs):
	"""A matplotlib Figure (no pyplot state) and its axes, styled."""
	from matplotlib.figure import Figure # not at import time: only needed for plots
	fig = Figure(figsize=(width, height), facecolor=SURFACE, layout='constrained')
	axes = fig.subplots(nrows, ncols, squeeze=False, **kwargs)
	for ax in axes.flat:
		ax.set_facecolor(SURFACE)
		for side in ('top', 'right'):
			ax.spines[side].set_visible(False)
		for side in ('left', 'bottom'):
			ax.spines[side].set_color(GRID)
		ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY, labelsize=8)
		ax.grid(True, color=GRID, linewidth=0.6)
		ax.set_axisbelow(True)
	return fig, axes


def _title(ax, text: str) -> None:
	ax.set_title(text, color=TEXT, fontsize=11, loc='left')


def _labels(ax, xlabel: str, ylabel: str) -> None:
	ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
	ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)


def _legend(ax) -> None:
	ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)


def _time_axis(times: np.ndarray) -> tuple[np.ndarray, str]:
	"""Times in ps, or in ns for long trajectories, with the axis label."""
	times = np.asarray(times, dtype=float)
	if len(times) and np.ptp(times) > 2000:
		return times / 1000.0, 'Time [ns]'
	return times, 'Time [ps]'


def _end_labels(ax, x: np.ndarray, series: np.ndarray, labels: list[str]) -> None:
	"""Direct labels at the line ends (<= 4 series), spread apart so they do not overlap."""
	if len(series) > 4 or not len(x):
		return
	low, high = ax.get_ylim()
	gap = 0.06 * (high - low)
	order = np.argsort(series[:, -1])
	positions = series[order, -1].astype(float)
	for i in range(1, len(positions)):
		positions[i] = max(positions[i], positions[i - 1] + gap)
	for y, k in zip(positions, order):
		ax.annotate(labels[k], (x[-1], y), xytext=(4, 0), textcoords='offset points', va='center',
		            fontsize=8, color=TEXT_SECONDARY, annotation_clip=False)


def _save(fig, fname: str | Path) -> Path:
	fname = Path(fname)
	fig.savefig(fname, dpi=DPI, facecolor=SURFACE, bbox_inches='tight')
	logger.debug("wrote plot %s", fname)
	return fname


def _default_labels(values: np.ndarray, labels: list[str] | None, noun: str) -> list[str]:
	return list(labels) if labels is not None else [f"{noun} {i}" for i in range(len(values))]


def plot_total(times: ArrayLike, total: ArrayLike, fname: str | Path, title: str = 'Total SASA') -> Path:
	"""Total SASA over time with its running (cumulative) mean, to judge convergence."""
	total = np.asarray(total, dtype=float)
	x, xlabel = _time_axis(times)
	fig, axes = _figure()
	ax = axes[0, 0]
	running = np.cumsum(total) / np.arange(1, len(total) + 1)
	ax.plot(x, total, color=SERIES_COLORS[0], linewidth=LINE_WIDTH, label='SASA')
	ax.plot(x, running, color=TEXT, linewidth=LINE_WIDTH, linestyle='--', label='running mean')
	_legend(ax)
	_labels(ax, xlabel, SASA_LABEL)
	if len(total):
		title = f"{title}: mean {total.mean():.3f} ± {total.std():.3f} nm²"
	_title(ax, title)
	return _save(fig, fname)


def plot_time(times: ArrayLike, values: ArrayLike, fname: str | Path, labels: list[str] | None = None,
              noun: str = 'molecule') -> Path:
	"""
	SASA of every series over time: up to 8 series as colored lines (direct labels for
	up to 4), more as faint lines behind their mean.
	"""
	values = np.atleast_2d(np.asarray(values, dtype=float))
	labels = _default_labels(values, labels, noun)
	x, xlabel = _time_axis(times)
	fig, axes = _figure()
	ax = axes[0, 0]
	if len(values) <= MAX_SERIES:
		for color, series, label in zip(SERIES_COLORS, values, labels):
			ax.plot(x, series, color=color, linewidth=LINE_WIDTH, label=label)
		if len(values) > 1:
			_legend(ax)
			_end_labels(ax, x, values, labels)
		_title(ax, f"SASA per {noun}" if len(values) > 1 else f"SASA of {labels[0]}")
	else:
		for series in values:
			ax.plot(x, series, color=MUTED, linewidth=0.5, alpha=0.4)
		ax.plot([], [], color=MUTED, linewidth=0.8, label=f"individual {noun}s")
		mean = values.mean(axis=0)
		ax.plot(x, mean, color=SERIES_COLORS[0], linewidth=LINE_WIDTH, label=f"mean of {len(values)} {noun}s")
		_legend(ax)
		_title(ax, f"SASA of {len(values)} {noun}s")
	_labels(ax, xlabel, SASA_LABEL)
	return _save(fig, fname)


def _pad(low: float, high: float, bin_width: float = BIN_WIDTH) -> float:
	"""Padding (nm^2) around the range ``low`` to ``high``: a fraction of it, at least a few bins."""
	return max(PAD_FRACTION * (high - low), MIN_PAD_BINS * bin_width)


def distribution(values: ArrayLike, bin_width: float = BIN_WIDTH, pad: float | None = None) -> np.ndarray:
	"""
	Probability density of SASA values as in fast_forward: a histogram on bins of
	``bin_width`` (nm^2) on the grid of its multiples (fast_forward: 0.01 nm^2 from 0 to
	50 nm^2; here the range of the values plus ``pad`` (nm^2) of empty bins on both sides,
	not below 0, so that the density drops to 0 at both ends, as with fast_forward's range).
	By default ``pad`` is 10 % of the range of the values, at least two bins.

	Returns
	-------
	np.ndarray
		(n_bins, 2) bin centre (nm^2) and probability density (nm^-2)
	"""
	values = np.asarray(values, dtype=float).ravel()
	values = values[np.isfinite(values)]
	if not len(values):
		return np.zeros((0, 2))
	if pad is None:
		pad = _pad(values.min(), values.max(), bin_width)
	first = max(0, int(np.floor((values.min() - pad) / bin_width))) # SASA >= 0, as fast_forward's bins
	last = int(np.floor((values.max() + pad) / bin_width)) + 1
	edges = np.arange(first, last + 1) * bin_width # integer grid: no drift of the bin edges
	density, edges = np.histogram(values, bins=edges, density=True)
	return np.column_stack((edges[:-1] + np.diff(edges) / 2.0, density))


def _zoom(ax, curves: list[np.ndarray], threshold: float = ZOOM_THRESHOLD) -> None:
	"""
	As fast_forward: limit the x axis to where any density exceeds ``threshold`` times its
	maximum, plus padding on both sides (fast_forward: 0.25 nm^2; here 10 % of that range,
	at least two bins, so that narrow distributions such as single atoms fill the plot),
	but not below 0.
	"""
	low, high = np.inf, -np.inf
	for curve in curves:
		if not len(curve) or curve[:, 1].max() <= 0:
			continue
		significant = curve[curve[:, 1] > threshold * curve[:, 1].max(), 0]
		low, high = min(low, significant.min()), max(high, significant.max())
	if np.isfinite(low):
		pad = _pad(low, high)
		ax.set_xlim(max(0.0, low - pad), high + pad) # SASA >= 0


def _plot_density(ax, curve: np.ndarray, color: str = DISTRIBUTION_COLOR, label: str | None = None) -> None:
	ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=LINE_WIDTH, label=label)


def plot_distribution(values: ArrayLike, fname: str | Path, labels: list[str] | None = None,
                      noun: str = 'molecule', bin_width: float = BIN_WIDTH) -> Path:
	"""
	SASA distribution (probability density, see :func:`distribution`) as lines, zoomed as
	in fast_forward: one series alone, up to 8 overlaid, more pooled into one.
	"""
	values = np.atleast_2d(np.asarray(values, dtype=float))
	labels = _default_labels(values, labels, noun)
	fig, axes = _figure()
	ax = axes[0, 0]
	if len(values) == 1 or len(values) > MAX_SERIES:
		curves = [distribution(values, bin_width)]
		_plot_density(ax, curves[0])
		_title(ax, f"{len(values)} {noun}s pooled sasa" if len(values) > 1 else f"{labels[0]} sasa")
	else:
		curves = [distribution(series, bin_width) for series in values]
		for color, curve, label in zip(SERIES_COLORS, curves, labels):
			_plot_density(ax, curve, color, label)
		_legend(ax)
		_title(ax, f"sasa per {noun}")
	_zoom(ax, curves)
	_labels(ax, SASA_LABEL, DENSITY_LABEL)
	return _save(fig, fname)


def plot_distribution_panels(values: ArrayLike | list[ArrayLike], fname: str | Path,
                             labels: list[str] | None = None, noun: str = 'molecule',
                             bin_width: float = BIN_WIDTH, max_panels: int = MAX_PANELS) -> Path:
	"""
	The SASA distribution of every series in its own panel, laid out as fast_forward's
	distribution plots: 5 columns of 4 x 4 inch panels titled by the label, each zoomed
	to its distribution, the density label on the first column.

	``values`` is (n_series, n_frames), or a list of series of any length (e.g. an atom
	pooled over the molecules of its type, see :func:`pooled_atom_sasa`).
	"""
	values = [np.asarray(series, dtype=float).ravel() for series in
	          (values if isinstance(values, list) else np.atleast_2d(np.asarray(values, dtype=float)))]
	labels = _default_labels(values, labels, noun)
	shown = values[:max_panels]
	ncols = min(5, len(shown))
	nrows = -(len(shown) // -ncols)
	fig, axes = _figure(width=4.0 * ncols, height=4.0 * nrows, nrows=nrows, ncols=ncols)
	for ax, series, label in zip(axes.flat, shown, labels):
		curve = distribution(series, bin_width)
		_plot_density(ax, curve)
		_zoom(ax, [curve])
		_title(ax, label)
		ax.set_xlabel(SASA_LABEL, color=TEXT_SECONDARY, fontsize=9)
	for ax in axes[:, 0]:
		ax.set_ylabel(DENSITY_LABEL, color=TEXT_SECONDARY, fontsize=9)
	for ax in list(axes.flat)[len(shown):]:
		fig.delaxes(ax)
	if len(values) > max_panels:
		fig.suptitle(f"first {max_panels} of {len(values)} {noun}s", color=TEXT_SECONDARY, fontsize=9,
		             x=0.01, ha='left')
	return _save(fig, fname)


def pooled_atom_sasa(sasa) -> tuple[list[str], list[np.ndarray]]:
	"""
	SASA of every atom of a molecule type, pooled over all molecules of that type: the
	first atom of every GSH together, the second together, ...

	Molecules are of one type if they have the same moltype (tpr), else the same atom and
	residue names in the same order. Without ``per_molecule`` (one body) every atom is on
	its own. Atoms without a radius (dummy beads, virtual sites: no SASA) are left out.
	Needs the per-atom SASA (``save_atoms=True``) of the last :meth:`SASA.run`.

	Parameters
	----------
	sasa : SASA

	Returns
	-------
	labels : list of str
		'<resname><resid>:<name>' of the atom in the first molecule of its type, with the
		type in front if there are several types
	pooled : list of np.ndarray
		(n_molecules_of_type * n_frames,) SASA (nm^2) of every atom of every type
	"""
	types = {} # type -> list of (n_atoms, n_frames) per-atom SASA of its molecules
	first = {} # type -> atoms of its first molecule
	for i in range(sasa.n_molecules):
		atoms = sasa.molecule_atoms(i)
		if hasattr(atoms, 'moltypes'):
			key = atoms.moltypes[0]
		else:
			key = (tuple(atoms.names), tuple(atoms.resnames) if hasattr(atoms, 'resnames') else ())
		types.setdefault(key, []).append(sasa.get_sasa(atoms, per_atom=True))
		first.setdefault(key, atoms)
	labels, pooled = [], []
	for key, molecules in types.items():
		atoms = first[key]
		prefix = f"{key if isinstance(key, str) else atoms.resnames[0]} " if len(types) > 1 else ''
		stacked = np.stack(molecules, axis=1) # (n_atoms, n_molecules, n_frames)
		radii = sasa.radii[sasa._lookup[atoms.indices]] # the same for all molecules of the type
		for atom, radius, series in zip(atoms, radii, stacked):
			if radius <= 0: # no SASA computed
				continue
			labels.append(f"{prefix}{atom.resname}{atom.resid}:{atom.name}")
			pooled.append(series.ravel())
	return labels, pooled


def save_plots(directory: str | Path, times: ArrayLike, values: ArrayLike, labels: list[str] | None = None,
               total: ArrayLike | None = None, kind: str = 'molecule', atoms: list[ArrayLike] | None = None,
               atom_labels: list[str] | None = None) -> list[Path]:
	"""
	Save the standard plots of a SASA time series in ``directory`` (created if needed).

	Parameters
	----------
	directory : str or Path
	times : array-like
		(n_frames,) times (ps)
	values : array-like
		(n_series, n_frames) SASA (nm^2) of the molecules or groups (for 'whole' the one body)
	labels : list of str, optional
		label of every series
	total : array-like, optional
		(n_frames,) total SASA (nm^2); default the sum of ``values``
	kind : str
		what the series are: 'molecule', 'group', or 'whole' (one series: the whole
		selection as one body)
	atoms : list of array-like, optional
		SASA of every atom, e.g. pooled over the molecules of its type (see
		:func:`pooled_atom_sasa`): adds ``sasa_distribution_per_atom.png``
	atom_labels : list of str, optional
		label of every atom

	Returns
	-------
	list of Path
		the written files:

		- ``sasa_total.png``: total SASA over time with its running mean
		- ``sasa_time.png``: every molecule/group over time (more than one series, or groups)
		- ``sasa_distribution.png``: distribution of the series (of the total for 'whole')
		- ``sasa_distribution_per_<kind>.png``: one panel per molecule/group (more than one)
		- ``sasa_distribution_per_atom.png``: one panel per atom (with ``atoms``)
	"""
	if kind not in ('molecule', 'group', 'whole'):
		raise ValueError(f"kind must be 'molecule', 'group' or 'whole', got {kind!r}.")
	directory = Path(directory)
	directory.mkdir(parents=True, exist_ok=True)
	values = np.atleast_2d(np.asarray(values, dtype=float))
	total = values.sum(axis=0) if total is None else np.asarray(total, dtype=float)
	files = [plot_total(times, total, directory / 'sasa_total.png',
	                    title='SASA of the selection' if kind == 'whole' else 'Total SASA')]
	if kind == 'whole':
		files.append(plot_distribution(total, directory / 'sasa_distribution.png', ['the selection']))
	else:
		if len(values) > 1 or kind == 'group':
			files.append(plot_time(times, values, directory / 'sasa_time.png', labels, noun=kind))
		files.append(plot_distribution(values, directory / 'sasa_distribution.png', labels, noun=kind))
		if len(values) > 1:
			files.append(plot_distribution_panels(values, directory / f'sasa_distribution_per_{kind}.png',
			                                      labels, noun=kind))
	if atoms is not None:
		files.append(plot_distribution_panels(list(atoms), directory / 'sasa_distribution_per_atom.png',
		                                      atom_labels, noun='atom', max_panels=MAX_ATOM_PANELS))
	logger.info("wrote %d plot(s) to %s", len(files), directory)
	return files
