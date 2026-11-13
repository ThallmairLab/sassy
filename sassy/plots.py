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
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike

from .groups import atom_label

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
HIGHLIGHT_COLOR = '#c00000' # the frame of the surface (red, dashed: apart from the series colors)
SASA_LABEL = 'SASA [nm²]'
DENSITY_LABEL = 'Probability Density'


@dataclass
class Highlight:
	"""
	A frame to highlight in the plots by a red vertical line, e.g. that of the surface
	(:meth:`SASA.write_surface`): at its time in the time plots and at its SASA in the
	distributions.

	Attributes
	----------
	time : float
		time (ps) of the frame
	sasa : float, optional
		SASA (nm^2) of the highlighted series in the frame; None: not in the distributions
	series : int, optional
		index of the highlighted series (its panel in the per-series distributions);
		None: in no panel
	label : str
		legend label of the line
	atoms : dict of int to float, optional
		SASA (nm^2) of the atoms in the frame by their panel in the per-atom distributions
		(see :func:`atom_sasa_at`); None: in no atom panel
	"""
	time: float
	sasa: float | None = None
	series: int | None = None
	label: str = 'surface frame'
	atoms: dict[int, float] | None = None


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


def _legend(ax, above: bool = False) -> None:
	"""Legend inside the axes, or with ``above`` in one row above it, right of the title."""
	if above:
		ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc='lower right',
		          bbox_to_anchor=(1.0, 1.0), ncols=len(ax.get_legend_handles_labels()[0]),
		          borderaxespad=0.2, handlelength=1.5, columnspacing=1.2)
	else:
		ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)


def _time_axis(times: np.ndarray) -> tuple[np.ndarray, str]:
	"""Times in ps, or in ns for long trajectories, with the axis label."""
	times = np.asarray(times, dtype=float)
	if len(times) and np.ptp(times) > 2000:
		return times / 1000.0, 'Time [ns]'
	return times, 'Time [ps]'


def _mark(ax, x: float, label: str | None = None) -> None:
	"""The highlighted frame: a red dashed vertical line at ``x`` (time or SASA)."""
	ax.axvline(x, color=HIGHLIGHT_COLOR, linewidth=1.2, linestyle='--', label=label, zorder=3)


Highlights = Highlight | list[Highlight] | None


def _highlights(highlight: Highlights) -> list[Highlight]:
	"""None, one or several highlights as a list."""
	if highlight is None:
		return []
	return [highlight] if isinstance(highlight, Highlight) else list(highlight)


def _has_dot(highlight: Highlight) -> bool:
	return highlight.sasa is not None and highlight.series is not None


def _mark_time(ax, times: np.ndarray, highlight: Highlights, dot: bool = False) -> None:
	"""
	The highlighted frames at their time: a red dashed vertical line, or with ``dot`` a red
	dot at the SASA on the line of the series (if both are known, else the vertical line).
	"""
	scale = 1000.0 if _time_axis(times)[1] == 'Time [ns]' else 1.0
	for h in _highlights(highlight):
		if dot and _has_dot(h):
			ax.plot([h.time / scale], [h.sasa], linestyle='none', marker='o', markersize=7, color=HIGHLIGHT_COLOR,
			        markeredgecolor=SURFACE, markeredgewidth=1.5, label=h.label, zorder=4)
		else:
			_mark(ax, h.time / scale, h.label)


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


def _type_groups(types: list[str] | None, n_series: int) -> dict[str, np.ndarray]:
	"""Series indices of every type (in order of first appearance); one group without types."""
	if types is None:
		return {'': np.arange(n_series)}
	if len(types) != n_series:
		raise ValueError(f"types must have one entry per series ({n_series}), got {len(types)}.")
	groups = {}
	for i, name in enumerate(types):
		groups.setdefault(name, []).append(i)
	return {name: np.array(indices) for name, indices in groups.items()}


def plot_total(times: ArrayLike, total: ArrayLike, fname: str | Path, title: str = 'Total SASA',
               highlight: Highlights = None) -> Path:
	"""
	Total SASA over time with its running (cumulative) mean, to judge convergence, and the
	time of the ``highlight`` frame(s).
	"""
	total = np.asarray(total, dtype=float)
	x, xlabel = _time_axis(times)
	fig, axes = _figure()
	ax = axes[0, 0]
	running = np.cumsum(total) / np.arange(1, len(total) + 1)
	ax.plot(x, total, color=SERIES_COLORS[0], linewidth=LINE_WIDTH, label='SASA')
	ax.plot(x, running, color=TEXT, linewidth=LINE_WIDTH, linestyle='--', label='running mean')
	_mark_time(ax, times, highlight)
	_legend(ax)
	_labels(ax, xlabel, SASA_LABEL)
	if len(total):
		title = f"{title}: mean {total.mean():.3f} ± {total.std():.3f} nm²"
	_title(ax, title)
	return _save(fig, fname)


def _draw_time(ax, times: np.ndarray, values: np.ndarray, labels: list[str], noun: str,
               highlight: Highlights, title: str | None = None, legend_above: bool = False) -> None:
	"""
	Series over time in one axes: up to 8 as colored lines (direct labels for up to 4),
	more as faint lines behind their mean; the ``highlight`` frame(s) as a dot on the series.
	"""
	x, _ = _time_axis(times)
	if len(values) <= MAX_SERIES:
		for color, series, label in zip(SERIES_COLORS, values, labels):
			ax.plot(x, series, color=color, linewidth=LINE_WIDTH, label=label)
		_mark_time(ax, times, highlight, dot=True)
		if len(values) > 1 or _highlights(highlight):
			_legend(ax, legend_above)
		if len(values) > 1:
			_end_labels(ax, x, values, labels)
		default = f"SASA per {noun}" if len(values) > 1 else f"SASA of {labels[0]}"
	else:
		for series in values:
			ax.plot(x, series, color=MUTED, linewidth=0.5, alpha=0.4)
		ax.plot([], [], color=MUTED, linewidth=0.8, label=f"individual {noun}s")
		mean = values.mean(axis=0)
		ax.plot(x, mean, color=SERIES_COLORS[0], linewidth=LINE_WIDTH, label=f"mean of {len(values)} {noun}s")
		_mark_time(ax, times, highlight, dot=True)
		_legend(ax, legend_above)
		default = f"SASA of {len(values)} {noun}s"
	_title(ax, title or default)


def plot_time(times: ArrayLike, values: ArrayLike, fname: str | Path, labels: list[str] | None = None,
              noun: str = 'molecule', highlight: Highlights = None,
              types: list[str] | None = None, max_panels: int = MAX_PANELS) -> Path:
	"""
	SASA of every series over time: up to 8 series as colored lines (direct labels for
	up to 4), more as faint lines behind their mean. With several types (``types``, the
	type of every series) one panel per type, stacked vertically on a shared time axis
	(the first ``max_panels`` types). The ``highlight`` frame(s) as a dot on their series (in
	the panel of its type), or a vertical line at the time if the SASA or series is not
	known (in every panel).
	"""
	values = np.atleast_2d(np.asarray(values, dtype=float))
	labels = _default_labels(values, labels, noun)
	groups = _type_groups(types, len(values))
	_, xlabel = _time_axis(times)
	if len(groups) == 1:
		fig, axes = _figure()
		_draw_time(axes[0, 0], times, values, labels, noun, highlight)
		_labels(axes[0, 0], xlabel, SASA_LABEL)
		return _save(fig, fname)

	shown = list(groups.items())[:max_panels]
	fig, axes = _figure(height=2.8 * len(shown), nrows=len(shown), sharex=True)
	for ax, (name, indices) in zip(axes[:, 0], shown):
		local = [] # dots only in the panel of their series, at the index there; lines in all
		for h in _highlights(highlight):
			if not _has_dot(h):
				local.append(h)
			elif len(inside := np.flatnonzero(indices == h.series)):
				local.append(replace(h, series=int(inside[0])))
		n = len(indices)
		_draw_time(ax, times, values[indices], [labels[i] for i in indices], noun, local,
		           title=f"{name} ({n} {noun}{'s' if n > 1 else ''})", legend_above=True)
		ax.set_ylabel(SASA_LABEL, color=TEXT_SECONDARY, fontsize=9)
	axes[-1, 0].set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
	if len(groups) > max_panels:
		fig.suptitle(f"first {max_panels} of {len(groups)} {noun} types", color=TEXT_SECONDARY, fontsize=9,
		             x=0.01, ha='left')
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


def _zoom(ax, curves: list[np.ndarray], threshold: float = ZOOM_THRESHOLD,
          include: float | list[float] | None = None) -> None:
	"""
	As fast_forward: limit the x axis to where any density exceeds ``threshold`` times its
	maximum, plus padding on both sides (fast_forward: 0.25 nm^2; here 10 % of that range,
	at least two bins, so that narrow distributions such as single atoms fill the plot),
	but not below 0. The range is extended to ``include`` (nm^2, e.g. a highlighted SASA).
	"""
	low, high = np.inf, -np.inf
	for curve in curves:
		if not len(curve) or curve[:, 1].max() <= 0:
			continue
		significant = curve[curve[:, 1] > threshold * curve[:, 1].max(), 0]
		low, high = min(low, significant.min()), max(high, significant.max())
	include = [] if include is None else np.atleast_1d(include)
	if len(include):
		low, high = min(low, np.min(include)), max(high, np.max(include))
	if np.isfinite(low):
		pad = _pad(low, high)
		ax.set_xlim(max(0.0, low - pad), high + pad) # SASA >= 0


def _plot_density(ax, curve: np.ndarray, color: str = DISTRIBUTION_COLOR, label: str | None = None) -> None:
	ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=LINE_WIDTH, label=label)


def plot_distribution(values: ArrayLike, fname: str | Path, labels: list[str] | None = None,
                      noun: str = 'molecule', bin_width: float = BIN_WIDTH,
                      highlight: Highlights = None) -> Path:
	"""
	SASA distribution (probability density, see :func:`distribution`) as lines, zoomed as
	in fast_forward: one series alone, up to 8 overlaid, more pooled into one; the SASA of
	the ``highlight`` frame(s) as a vertical line. (Several molecule types: see
	:func:`save_plots`, one panel per type.)
	"""
	marked = [h for h in _highlights(highlight) if h.sasa is not None]
	values = np.atleast_2d(np.asarray(values, dtype=float))
	labels = _default_labels(values, labels, noun)
	fig, axes = _figure()
	ax = axes[0, 0]
	if len(values) == 1 or len(values) > MAX_SERIES:
		curves = [distribution(values, bin_width)]
		_plot_density(ax, curves[0])
		_title(ax, f"{len(values)} {noun}s pooled sasa" if len(values) > 1 else f"{labels[0]} sasa")
		for h in marked:
			_mark(ax, h.sasa, h.label)
		if marked:
			_legend(ax)
	else:
		curves = [distribution(series, bin_width) for series in values]
		for color, curve, label in zip(SERIES_COLORS, curves, labels):
			_plot_density(ax, curve, color, label)
		for h in marked:
			_mark(ax, h.sasa, h.label)
		_legend(ax)
		_title(ax, f"sasa per {noun}")
	_zoom(ax, curves, include=[h.sasa for h in marked])
	_labels(ax, SASA_LABEL, DENSITY_LABEL)
	return _save(fig, fname)


def plot_distribution_panels(values: ArrayLike | list[ArrayLike], fname: str | Path,
                             labels: list[str] | None = None, noun: str = 'molecule',
                             bin_width: float = BIN_WIDTH, max_panels: int = MAX_PANELS,
                             marks: dict[int, float | list[tuple[float, str]]] | None = None,
                             mark_label: str = 'surface frame', colors: list[str] | None = None) -> Path:
	"""
	The SASA distribution of every series in its own panel, laid out as fast_forward's
	distribution plots: 5 columns of 4 x 4 inch panels titled by the label, each zoomed
	to its distribution, the density label on the first column.

	``values`` is (n_series, n_frames), or a list of series of any length (e.g. an atom
	pooled over the molecules of its type, see :func:`pooled_atom_sasa`). ``marks`` (panel
	index -> SASA in nm^2 labelled ``mark_label``, or a list of (SASA, label), e.g. of
	highlighted frames) are drawn as red dashed vertical lines; a label is in the legend of
	the first panel it marks. ``colors``: the line color of every panel (default all the same).
	"""
	marks = {i: [(m, mark_label)] if np.isscalar(m) else list(m) for i, m in (marks or {}).items()}
	shown_labels = set() # every label in a legend once
	values = [np.asarray(series, dtype=float).ravel() for series in
	          (values if isinstance(values, list) else np.atleast_2d(np.asarray(values, dtype=float)))]
	labels = _default_labels(values, labels, noun)
	shown = values[:max_panels]
	ncols = min(5, len(shown))
	nrows = -(len(shown) // -ncols)
	fig, axes = _figure(width=4.0 * ncols, height=4.0 * nrows, nrows=nrows, ncols=ncols)
	colors = colors or [DISTRIBUTION_COLOR] * len(values)
	for i, (ax, series, label) in enumerate(zip(axes.flat, shown, labels)):
		curve = distribution(series, bin_width)
		_plot_density(ax, curve, colors[i])
		new = False
		for sasa, legend_label in marks.get(i, []):
			_mark(ax, sasa, None if legend_label in shown_labels else legend_label)
			new |= legend_label not in shown_labels
			shown_labels.add(legend_label)
		if new:
			_legend(ax)
		_zoom(ax, [curve], include=[sasa for sasa, _ in marks.get(i, [])])
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


def _with_radius(sasa, molecule: int) -> np.ndarray:
	"""Mask of the atoms of a molecule with a radius (with SASA; no dummy beads, virtual sites)."""
	return sasa.radii[sasa._members[molecule]] > 0


def pooled_atom_sasa(sasa) -> tuple[list[str], list[np.ndarray]]:
	"""
	SASA of every atom of a molecule type, pooled over all molecules of that type: the
	first atom of every GSH together, the second together, ...

	Molecules are of one type (:attr:`SASA.types`) if they have the same moltype (tpr), else
	the same atom and residue names in the same order (with ``bodies='residue'``: residues of one
	residue type); with ``bodies='selection'`` (one body) every atom is on its own. Atoms without a radius (dummy beads, virtual sites: no SASA) are left out.
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
	sasa._check_run(atoms=True)
	labels, pooled = [], []
	for name, molecules in sasa.types.items():
		members = np.stack([sasa._members[i] for i in molecules], axis=1) # (n_atoms, n_molecules) indices in ag
		prefix = f"{name} " if len(sasa.types) > 1 else ''
		kept = _with_radius(sasa, molecules[0]) # the same for all molecules of the type
		stacked = sasa.sasa[members[kept]] # (n_kept, n_molecules, n_frames)
		for atom, series in zip(sasa.ag[members[kept, 0]], stacked):
			labels.append(f"{prefix}{atom_label(atom)}")
			pooled.append(series.ravel())
	return labels, pooled


def atom_sasa_at(sasa, molecule: int, frame: int) -> dict[int, float]:
	"""
	SASA of the atoms of one molecule in one frame, by their panel in the per-atom
	distributions of :func:`pooled_atom_sasa` (e.g. to highlight the frame of a surface).

	Parameters
	----------
	sasa : SASA
	molecule : int
		molecule index (negative counts from the end)
	frame : int
		trajectory frame; must be one of the last :meth:`SASA.run`

	Returns
	-------
	dict of int to float
		panel index -> SASA (nm^2); empty if the frame was not analysed
	"""
	sasa._check_run(atoms=True)
	analysed = np.flatnonzero(sasa.frames == frame)
	if not len(analysed):
		return {}
	molecule %= sasa.n_molecules
	offset = 0
	for molecules in sasa.types.values():
		kept = _with_radius(sasa, molecules[0])
		if molecule in molecules:
			values = sasa.sasa[sasa._members[molecule][kept], analysed[0]]
			return {offset + k: float(value) for k, value in enumerate(values)}
		offset += int(kept.sum())
	raise AssertionError('unreachable: every molecule has a type')


def save_plots(directory: str | Path, times: ArrayLike, values: ArrayLike, labels: list[str] | None = None,
               total: ArrayLike | None = None, kind: str = 'molecule', atoms: list[ArrayLike] | None = None,
               atom_labels: list[str] | None = None, highlight: Highlights = None,
               types: list[str] | None = None) -> list[Path]:
	"""
	Save the standard plots of a SASA time series in ``directory`` (created if needed).

	Parameters
	----------
	directory : str or Path
	times : array-like
		(n_frames,) times (ps)
	values : array-like
		(n_series, n_frames) SASA (nm^2) of the molecules, residues or groups (for 'whole' the one body)
	labels : list of str, optional
		label of every series
	total : array-like, optional
		(n_frames,) total SASA (nm^2); default the sum of ``values``
	kind : str
		what the series are: 'molecule', 'residue', 'group', or 'whole' (one series: the
		whole selection as one body)
	atoms : list of array-like, optional
		SASA of every atom, e.g. pooled over the molecules of its type (see
		:func:`pooled_atom_sasa`): adds ``sasa_distribution_per_atom.png``
	atom_labels : list of str, optional
		label of every atom
	highlight : Highlight or list of Highlight, optional
		frame(s) to mark in red (e.g. those of the surfaces, one per type): a vertical line at its time in
		``sasa_total``, a dot on its series in ``sasa_time`` (a line if its SASA is not
		known); a vertical line at its SASA (``highlight.sasa``, of the series
		``highlight.series``) in ``sasa_distribution`` and the panel of that series, and at
		the SASA of every atom (``highlight.atoms``) in its per-atom panel
	types : list of str, optional
		type of every series (e.g. :attr:`SASA.molecule_types`): with several types,
		``sasa_distribution`` has one panel per type (pooled over its series, each zoomed to
		its range), ``sasa_time`` one panel per type stacked vertically (the first 25 types
		each), and the per-series panels are ordered and colored by type

	Returns
	-------
	list of Path
		the written files:

		- ``sasa_total.png``: total SASA over time with its running mean
		- ``sasa_time.png``: every molecule/residue/group over time (more than one series, or groups)
		- ``sasa_distribution.png``: distribution of the series (of the total for 'whole')
		- ``sasa_distribution_per_<kind>.png``: one panel per molecule/residue/group (more than one)
		- ``sasa_distribution_per_atom.png``: one panel per atom (with ``atoms``)
	"""
	if kind not in ('molecule', 'residue', 'group', 'whole'):
		raise ValueError(f"kind must be 'molecule', 'residue', 'group' or 'whole', got {kind!r}.")
	directory = Path(directory)
	directory.mkdir(parents=True, exist_ok=True)
	values = np.atleast_2d(np.asarray(values, dtype=float))
	total = values.sum(axis=0) if total is None else np.asarray(total, dtype=float)
	highlights = _highlights(highlight)
	dots = [h for h in highlights if _has_dot(h)]

	def panel_marks(panel_of: dict[int, int]) -> dict[int, list[tuple[float, str]]]:
		"""Marks of the highlights by panel; ``panel_of``: series -> panel."""
		marks = {}
		for h in dots:
			marks.setdefault(panel_of[h.series], []).append((h.sasa, h.label))
		return marks

	files = [plot_total(times, total, directory / 'sasa_total.png',
	                    title='SASA of the selection' if kind == 'whole' else 'Total SASA', highlight=highlight)]
	if kind == 'whole':
		files.append(plot_distribution(total, directory / 'sasa_distribution.png', ['the selection'],
		                               highlight=highlight))
	else:
		if len(values) > 1 or kind == 'group':
			files.append(plot_time(times, values, directory / 'sasa_time.png', labels, noun=kind,
			                       highlight=highlight, types=types))
		groups = _type_groups(types, len(values))
		type_colors = SERIES_COLORS if len(groups) <= MAX_SERIES else [DISTRIBUTION_COLOR] * len(groups)
		if len(groups) > 1: # one panel per type, pooled over its series
			type_of = {i: k for k, indices in enumerate(groups.values()) for i in indices}
			files.append(plot_distribution_panels(
				[values[indices].ravel() for indices in groups.values()], directory / 'sasa_distribution.png',
				[f"{name} ({len(indices)} {kind}{'s' if len(indices) > 1 else ''})" for name, indices in groups.items()],
				noun=f'{kind} type', marks=panel_marks(type_of), colors=list(type_colors[:len(groups)])))
		else:
			files.append(plot_distribution(values, directory / 'sasa_distribution.png', labels, noun=kind,
			                               highlight=highlight))
		if len(values) > 1:
			# panels ordered by type (stable: in their order within a type), colored by type
			order = np.concatenate(list(groups.values()))
			colors = None
			if 1 < len(groups) <= MAX_SERIES:
				colors = [color for color, indices in zip(SERIES_COLORS, groups.values()) for _ in indices]
			labels = _default_labels(values, labels, kind)
			panel_of = {int(series): panel for panel, series in enumerate(order)}
			files.append(plot_distribution_panels(values[order], directory / f'sasa_distribution_per_{kind}.png',
			                                      [labels[i] for i in order], noun=kind, marks=panel_marks(panel_of),
			                                      colors=colors))
	if atoms is not None:
		atom_marks = {}
		for h in highlights:
			for panel, sasa in (h.atoms or {}).items():
				atom_marks.setdefault(panel, []).append((sasa, h.label))
		files.append(plot_distribution_panels(list(atoms), directory / 'sasa_distribution_per_atom.png',
		                                      atom_labels, noun='atom', max_panels=MAX_ATOM_PANELS,
		                                      marks=atom_marks))
	logger.info("wrote %d plot(s) to %s", len(files), directory)
	return files
