"""
Command line interface: SASA of the molecules, atoms or selections of a
trajectory, written as an xvg (text) or npz file.

	sassy -s topol.tpr -f traj.xtc -select 'moltype GSH' -o sasa.xvg
"""
import argparse
import logging
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
import numpy as np
import MDAnalysis as mda
from MDAnalysis.exceptions import SelectionError

from .. import __version__
from ..groups import atom_label
from ..radii import PROBE_RADIUS
from ..sasa import N_DOTS, SASA
from ..surface import STATISTICS, SURFACE_DOTS

logger = logging.getLogger('sassy')


class _Formatter(logging.Formatter):
	"""Info messages plain, others as ``sassy: WARNING: ...``; with ``debug`` as ``time level [module] message``."""
	def __init__(self, debug: bool = False) -> None:
		super().__init__()
		self.debug = debug

	def format(self, record: logging.LogRecord) -> str:
		message = record.getMessage()
		if self.debug:
			message = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:7} [{record.name}] {message}"
		elif record.levelno != logging.INFO:
			message = f"sassy: {record.levelname}: {message}"
		if record.exc_info:
			message += '\n' + self.formatException(record.exc_info)
		return message


def _setup_logging(verbose: bool, quiet: bool) -> None:
	"""Log the sassy messages to stderr: warnings only (quiet), info (default) or debug (verbose)."""
	level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
	for handler in logger.handlers[:]: # main() may run several times (tests, scripts)
		logger.removeHandler(handler)
	handler = logging.StreamHandler(sys.stderr)
	handler.setFormatter(_Formatter(debug=verbose))
	logger.addHandler(handler)
	logger.setLevel(level)
	logger.propagate = False


def _radius(text: str) -> tuple[str, float]:
	"""``SYMBOL=RADIUS`` -> (symbol, radius in nm)."""
	symbol, sep, value = text.partition('=')
	try:
		if not sep or not symbol:
			raise ValueError
		return symbol.strip(), float(value)
	except ValueError:
		raise argparse.ArgumentTypeError(f"expected SYMBOL=RADIUS (nm), e.g. Na=0.227, got {text!r}") from None


def _positive_int(text: str) -> int:
	"""An integer >= 1."""
	try:
		value = int(text)
	except ValueError:
		raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
	if value < 1:
		raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value}")
	return value


def _non_negative(text: str) -> float:
	"""A number >= 0."""
	try:
		value = float(text)
	except ValueError:
		raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
	if not value >= 0:
		raise argparse.ArgumentTypeError(f"expected a number >= 0, got {text}")
	return value


def _statistic(text: str) -> str | float:
	"""Name of a statistic of the SASA series, or a SASA value (nm^2)."""
	if text in STATISTICS:
		return text
	try:
		return float(text)
	except ValueError:
		raise argparse.ArgumentTypeError(f"expected {', '.join(STATISTICS)} or a SASA (nm^2), got {text!r}") from None


def parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog='sassy',
		description="Shrake-Rupley solvent accessible surface area (nm^2) of a selection over a "
		            "trajectory. By default every molecule is taken on its own (other molecules "
		            "do not bury it) and gets its own column; with whole the selection is one "
		            "body (its molecules bury each other) and one column is written. Columns "
		            "per atom (per-atom) or per group (group) instead.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument('-V', action='version', version=f'%(prog)s {__version__}')

	io = p.add_argument_group('input/output')
	io.add_argument('-s', required=True, help='topology (tpr, gro, pdb, ...); a tpr gives molecules and elements', dest='topology')
	io.add_argument('-f', nargs='*', default=[], help='trajectory file(s) (xtc, trr, ...); default the topology coordinates', dest='trajectory')
	io.add_argument('-o', default='sasa.xvg', help='output file: .xvg or .npz', dest='output')
	io.add_argument('-plots', nargs='?', const='.', default=None, metavar='DIR',
	                help='save plots (PNG) to this directory, the current one if none is given: total SASA '
	                     'over time with running mean, every molecule/group over time, distributions')

	sel = p.add_argument_group('selection')
	sel.add_argument('-select', default='all', help='MDAnalysis selection of the atoms whose SASA is computed')
	sel.add_argument('-whole', action='store_true', help='SASA of the whole selection as one body (its molecules bury each other), '
	                 'one column instead of one per molecule (each on its own)')
	sel.add_argument('-per-atom', action='store_true', help='one column per atom instead of per molecule/selection')
	sel.add_argument('-group', action='append', default=[], metavar='SELECTION',
	                 help='one column with the summed SASA of these atoms (MDAnalysis selection within select); repeatable')
	sel.add_argument('-total', action='store_true', help='add a column with the total SASA of the selection')

	frames = p.add_argument_group('frames', 'frame indices (0 = first; not times in ps as in GROMACS), '
	                              'as in Python slicing trajectory[b:e:skip]')
	frames.add_argument('-b', type=int, default=None, help='first frame (included; negative counts from the end)', dest='start')
	frames.add_argument('-e', type=int, default=None, help='stop frame (excluded: -b 0 -e 10 gives frames 0-9; '
	                    'negative counts from the end); default up to the last frame', dest='stop')
	frames.add_argument('-skip', type=_positive_int, default=None, help='use every skip-th frame', dest='step')

	model = p.add_argument_group('model')
	model.add_argument('-cg', action='store_true', help='Martini CG bead radii instead of atomistic radii')
	model.add_argument('-probe', type=_non_negative, default=PROBE_RADIUS, help='probe radius (nm)')
	model.add_argument('-ndots', type=_positive_int, default=N_DOTS, help='minimum number of dots per atom')
	model.add_argument('-radius', type=_radius, action='append', default=[], metavar='SYMBOL=RADIUS',
	                   help='radius (nm) by element (bead size with cg), adds to or replaces the defaults; repeatable')

	surf = p.add_argument_group('surface', 'structure and solvent accessible surface dots (Connolly-style '
	                            'dot surface) of one frame as PDB: atoms as ATOM, dots as HETATM records named DOT with the residue of their atom')
	surf.add_argument('-surface', nargs='?', const='.', default=None, metavar='DIR',
	                  help='write the surface of a representative molecule of every molecule type to DIR/<type>_sasa.pdb '
	                       '(e.g. GSH_sasa.pdb; the current directory if no DIR is given); with -whole that of the '
	                       'whole selection (selection_sasa.pdb)')
	surf.add_argument('-surface-molecule', default=None, metavar='MOLECULE',
	                  help='instead: only the surface of this molecule, by index (0 = first) or column label (e.g. '
	                       'GSH2), written to <label>_sasa.pdb')
	surface_frame = surf.add_mutually_exclusive_group()
	surface_frame.add_argument('-surface-frame', type=int, default=None, metavar='FRAME',
	                           help='trajectory frame of the surfaces (of every type the molecule whose SASA in this '
	                                'frame is closest to the median of the type); default see -surface-stat')
	surface_frame.add_argument('-surface-ref', default=None, metavar='FILE',
	                           help='reference structure (pdb, gro, ...; same atoms in the same order, e.g. a mapped '
	                                'atomistic structure for a CG model): use the analysed frame with the lowest RMSD '
	                                'to it, superposed onto it (one molecule type, or with -surface-molecule)')
	surface_frame.add_argument('-surface-stat', type=_statistic, default='median', metavar='STAT',
	                           help=f"of every type use the molecule and analysed frame whose SASA is closest to this "
	                                f"statistic of the SASA distribution of the type (all its molecules and frames; "
	                                f"{', '.join(STATISTICS)}; default median) or to this SASA (nm^2)")
	surf.add_argument('-surface-ndots', type=_positive_int, default=SURFACE_DOTS, help='minimum number of dots per atom of the surface')

	perf = p.add_argument_group('performance')
	perf.add_argument('-max-frame-atoms', type=_positive_int, default=20000, help='atoms (summed over frames) per parallel call')
	perf.add_argument('-nt', type=_positive_int, default=None, help='number of threads (default: all cores)', dest='threads')
	log = p.add_argument_group('logging').add_mutually_exclusive_group()
	log.add_argument('-v', action='store_true', help='verbose: debug messages (molecules, radii, timing)', dest='verbose')
	log.add_argument('-q', action='store_true', help='quiet: only warnings and errors, no progress bar', dest='quiet')
	return p


def _columns(sasa: SASA, args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
	"""(n_columns, n_frames) SASA and the column labels."""
	columns, labels = [], []
	if args.group:
		for group in args.group:
			columns.append(sasa.get_sasa(group))
			labels.append(group)
	elif args.per_atom:
		columns.extend(sasa.sasa)
		labels.extend(atom_label(a) for a in sasa.ag)
	elif args.whole: # one column: the total already
		columns.append(sasa.get_sasa())
		labels.append(args.select)
		return np.array(columns).reshape(1, len(sasa.frames)), labels
	else:
		columns.extend(sasa.get_molecule_sasa())
		labels.extend(sasa.molecule_labels)
	if args.total:
		columns.append(sasa.get_sasa())
		labels.append('total')
	n_frames = len(sasa.frames)
	return np.array(columns).reshape(len(columns), n_frames), labels


def _surface_file(directory: str, name: str) -> Path:
	"""DIR/<name>_sasa.pdb, with characters that do not belong in file names replaced."""
	safe = ''.join(c if c.isalnum() or c in '-_.+' else '_' for c in name)
	return Path(directory) / f"{safe}_sasa.pdb"


def _surface_options(args: argparse.Namespace, sasa: SASA) -> list[dict] | bool:
	"""
	The surfaces of -surface: one per molecule type (its representative molecule), or the
	molecule of -surface-molecule, as keyword arguments of :meth:`SASA.write_surface` with
	``fname``; checked before the (long) calculation. False (after logging the error) if invalid.
	"""
	if Path(args.surface).suffix.lower() == '.pdb':
		logger.error("-surface takes a directory (the surfaces are written to DIR/<type>_sasa.pdb), got %s",
		             args.surface)
		return False
	common = {'frame': args.surface_frame, 'statistic': args.surface_stat, 'ndots': args.surface_ndots}
	if args.surface_molecule is not None:
		if args.whole:
			logger.error("-surface-molecule cannot be combined with -whole (the selection is one body)")
			return False
		labels = sasa.molecule_labels
		text = args.surface_molecule
		if text.lstrip('-').isdigit():
			molecule = int(text)
			if not -len(labels) <= molecule < len(labels):
				logger.error("-surface-molecule %d out of range for %d molecule(s)", molecule, len(labels))
				return False
			molecule %= len(labels)
		elif text in labels:
			molecule = labels.index(text)
		else:
			logger.error("-surface-molecule %r is no molecule of the selection (labels %s)", text,
			             ', '.join(labels[:5]) + (', ...' if len(labels) > 5 else ''))
			return False
		logger.debug("surface of molecule %d (%s)", molecule, labels[molecule])
		jobs = [dict(common, molecule=molecule, fname=_surface_file(args.surface, labels[molecule]))]
	else:
		jobs = [dict(common, molecule_type=name, fname=_surface_file(args.surface, name)) for name in sasa.types]

	if args.surface_ref is not None:
		if len(jobs) > 1:
			logger.error("-surface-ref needs a single surface: the selection holds %d molecule types (%s); "
			             "give -surface-molecule", len(jobs), ', '.join(sasa.types))
			return False
		try:
			reference = mda.Universe(args.surface_ref).atoms
		except Exception as error: # missing file, unknown format, ...
			logger.error("cannot read the reference structure %s: %s", args.surface_ref, error)
			return False
		job = jobs[0]
		molecule = job['molecule'] if 'molecule' in job else int(sasa.types[job['molecule_type']][0])
		n_atoms = sasa.molecule_atoms(molecule).n_atoms
		if reference.n_atoms != n_atoms:
			logger.error("the reference structure %s has %d atoms, but the surface (%s) has %d", args.surface_ref,
			             reference.n_atoms, 'whole selection' if args.whole else sasa.molecule_labels[molecule],
			             n_atoms)
			return False
		job['reference'] = reference
	n_frames = sasa.ag.universe.trajectory.n_frames
	if args.surface_frame is not None and not -n_frames <= args.surface_frame < n_frames:
		logger.error("-surface-frame %d out of range for %d frame(s)", args.surface_frame, n_frames)
		return False
	return jobs


def _highlight(sasa: SASA, args: argparse.Namespace, surface):
	"""
	The frame of the surface for the plots: its time, and, if the frame was analysed, the
	SASA of its molecule (of the selection with -whole) and with -per-atom of its atoms.
	Groups are no molecules: time only.
	"""
	from ..plots import Highlight, atom_sasa_at
	series = surface.molecule
	label = f"surface frame {surface.frame}"
	if not args.whole:
		label += f" ({sasa.molecule_labels[series]})"
	analysed = np.flatnonzero(sasa.frames == surface.frame)
	if args.group or not len(analysed):
		if not len(analysed):
			logger.debug("surface frame %d was not analysed: marked in the time plots only", surface.frame)
		return Highlight(time=surface.time, label=label)
	return Highlight(time=surface.time, sasa=float(sasa.get_molecule_sasa(series)[analysed[0]]),
	                 series=series, label=label,
	                 atoms=atom_sasa_at(sasa, series, surface.frame) if args.per_atom else None)


def _write_xvg(path: Path, times: np.ndarray, columns: np.ndarray, labels: list[str], command: str) -> None:
	with open(path, 'w') as out:
		out.write(f"# created by sassy {__version__}\n# command: {command}\n")
		out.write('@    title "Solvent accessible surface"\n')
		out.write('@    xaxis  label "Time (ps)"\n')
		out.write('@    yaxis  label "Area (nm\\S2\\N)"\n')
		out.write('@TYPE xy\n')
		for i, label in enumerate(labels):
			label = label.replace('"', "'") # a double quote would end the legend string
			out.write(f'@ s{i} legend "{label}"\n')
		np.savetxt(out, np.column_stack([times, columns.T]), fmt='%12.4f')


def main(argv: Sequence[str] | None = None) -> int:
	p = parser()
	args = p.parse_args(argv)
	if args.per_atom and args.group:
		p.error('per-atom and group cannot be combined')
	if args.threads is not None:
		import numba
		if args.threads > numba.config.NUMBA_NUM_THREADS:
			p.error(f"-nt {args.threads}: at most {numba.config.NUMBA_NUM_THREADS} threads available")
		numba.set_num_threads(args.threads)
	_setup_logging(args.verbose, args.quiet)
	logger.debug("sassy %s: %s", __version__, ' '.join(sys.argv[1:] if argv is None else argv))
	output = Path(args.output)
	if output.suffix.lower() not in ('.xvg', '.npz'):
		logger.warning("output %s: unknown extension, written as xvg (text); use .xvg or .npz", output)

	logger.debug("reading %s %s", args.topology, ' '.join(args.trajectory))
	try:
		universe = mda.Universe(args.topology, *args.trajectory)
	except Exception as error: # missing file, unknown format, mismatching atoms, ...
		logger.error("cannot read %s: %s", ' '.join([args.topology, *args.trajectory]), error)
		return 1
	if not hasattr(universe, 'trajectory'):
		logger.error("%s has no coordinates; give a trajectory with -f", args.topology)
		return 1
	logger.debug("universe: %d atoms, %d frame(s)", universe.atoms.n_atoms, universe.trajectory.n_frames)
	try:
		ag = universe.select_atoms(args.select)
		if not ag.n_atoms:
			logger.error("the selection %r contains no atoms", args.select)
			return 1
		for group in args.group: # check the groups before the (long) calculation
			n_group = ag.select_atoms(group).n_atoms
			if not n_group:
				logger.error("the group %r contains no atoms of the selection", group)
				return 1
			logger.debug("group %r: %d atom(s)", group, n_group)
	except SelectionError as error:
		logger.error("invalid selection: %s", error)
		return 1

	try:
		sasa = SASA(ag, cg=args.cg, probe=args.probe, ndots=args.ndots, radii=dict(args.radius) or None,
		            max_frame_atoms=args.max_frame_atoms, per_molecule=not args.whole,
		            save_atoms=bool(args.per_atom or args.group)) # per-atom SASA only when needed
	except ValueError as error: # e.g. unknown elements
		logger.error("%s", error)
		return 1
	if not args.whole and len(sasa.types) > 1:
		logger.warning("the selection %r contains %d different molecule types; their molecules are written "
		               "as columns side by side (whole: one column for the selection)", args.select, len(sasa.types))
	surface = _surface_options(args, sasa) if args.surface else None
	if surface is False:
		return 1
	logger.info("%d atoms in %d %s, %d frame(s)", ag.n_atoms, sasa.n_molecules,
	            'body (whole selection)' if args.whole else 'molecule(s)',
	            len(universe.trajectory[args.start:args.stop:args.step]))
	sasa.run(start=args.start, stop=args.stop, step=args.step, verbose=not args.quiet)

	columns, labels = _columns(sasa, args)
	output.parent.mkdir(parents=True, exist_ok=True)
	if output.suffix.lower() == '.npz':
		with open(output, 'wb') as out: # a file object: np.savez would append .npz to e.g. x.NPZ
			np.savez(out, times=sasa.times, frames=sasa.frames, sasa=columns, labels=np.array(labels))
	else:
		command = 'sassy ' + shlex.join(sys.argv[1:] if argv is None else argv)
		_write_xvg(output, sasa.times, columns, labels, command)
	logger.info("wrote %d column(s) x %d frame(s) to %s", len(labels), len(sasa.frames), output)

	status, highlights = 0, []
	if surface is not None:
		Path(args.surface).mkdir(parents=True, exist_ok=True)
		for job in surface:
			try:
				result = sasa.write_surface(return_surface=True, **job)
			except (IndexError, KeyError, ValueError) as error: # e.g. frame out of range
				logger.error("surface %s: %s", job['fname'], error)
				status = 1 # the other surfaces and the plots are still written
			else:
				highlights.append(_highlight(sasa, args, result))

	if args.plots is not None:
		from ..plots import pooled_atom_sasa, save_plots
		kind = 'group' if args.group else 'whole' if args.whole else 'molecule'
		if args.group:
			has_total = args.total # the last column is the total
			values, series_labels = (columns[:-1], labels[:-1]) if has_total else (columns, labels)
		elif args.whole:
			values, series_labels = sasa.get_sasa()[None], [args.select]
		else: # molecules, also with -per-atom: the same plots as without
			values, series_labels = sasa.get_molecule_sasa(), sasa.molecule_labels
		atom_labels, atoms = pooled_atom_sasa(sasa) if args.per_atom else (None, None)
		save_plots(args.plots, sasa.times, values, series_labels, total=sasa.get_sasa(), kind=kind,
		           atoms=atoms, atom_labels=atom_labels, highlight=highlights,
		           types=sasa.molecule_types if kind == 'molecule' else None)
	return status


if __name__ == '__main__':
	sys.exit(main())
