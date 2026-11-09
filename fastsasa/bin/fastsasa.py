"""
Command line interface: SASA of the molecules, atoms or selections of a
trajectory, written as an xvg (text) or npz file.

	fastsasa -s topol.tpr -f traj.xtc select 'moltype GSH' -o sasa.xvg
"""
import argparse
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
import numpy as np

from .. import __version__
from ..radii import PROBE_RADIUS
from ..sasa import N_DOTS, SASA


def _radius(text: str) -> tuple[str, float]:
	"""``SYMBOL=RADIUS`` -> (symbol, radius in nm)."""
	symbol, sep, value = text.partition('=')
	try:
		if not sep or not symbol:
			raise ValueError
		return symbol.strip(), float(value)
	except ValueError:
		raise argparse.ArgumentTypeError(f"expected SYMBOL=RADIUS (nm), e.g. Na=0.227, got {text!r}") from None


def parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog='fastsasa',
		description="Shrake-Rupley solvent accessible surface area (nm^2) of every molecule of a "
		            "selection over a trajectory. Every molecule is taken on its own (other "
		            "molecules do not bury it). Writes one column per molecule, per atom "
		            "(per-atom) or per group (group).",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument('-V', action='version', version=f'%(prog)s {__version__}')

	io = p.add_argument_group('input/output')
	io.add_argument('-s', required=True, help='topology (tpr, gro, pdb, ...); a tpr gives molecules and elements', dest='topology')
	io.add_argument('-f', nargs='*', default=[], help='trajectory file(s) (xtc, trr, ...); default the topology coordinates', dest='trajectory')
	io.add_argument('-o', default='sasa.xvg', help='output file: .xvg/.dat (text) or .npz (times, frames, sasa, labels)', dest='output')

	sel = p.add_argument_group('selection')
	sel.add_argument('-select', default='all', help='MDAnalysis selection of the atoms whose SASA is computed')
	sel.add_argument('-per-atom', action='store_true', help='one column per atom instead of per molecule')
	sel.add_argument('-group', action='append', default=[], metavar='SELECTION',
	                 help='one column with the summed SASA of these atoms (MDAnalysis selection within select); repeatable')
	sel.add_argument('-total', action='store_true', help='add a column with the total SASA of the selection')

	frames = p.add_argument_group('frames')
	frames.add_argument('-b', type=int, default=None, help='First frame', dest='start')
	frames.add_argument('-e', type=int, default=None, help='Last frame', dest='stop')
	frames.add_argument('-skip', type=int, default=None, help='use every skip-th frame', dest='step')

	model = p.add_argument_group('model')
	model.add_argument('-cg', action='store_true', help='Martini CG bead radii instead of atomistic radii')
	model.add_argument('-probe', type=float, default=PROBE_RADIUS, help='probe radius (nm)')
	model.add_argument('-ndots', type=int, default=N_DOTS, help='minimum number of dots per atom')
	model.add_argument('-radius', type=_radius, action='append', default=[], metavar='SYMBOL=RADIUS',
	                   help='radius (nm) by element (bead size with cg), adds to or replaces the defaults; repeatable')

	perf = p.add_argument_group('performance')
	perf.add_argument('-max-frame-atoms', type=int, default=20000, help='atoms (summed over frames) per parallel call')
	perf.add_argument('-nt', type=int, default=None, help='number of threads (default: all cores)', dest='threads')
	perf.add_argument('-q', action='store_true', help='no progress bar', dest='quiet')
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
		labels.extend(f"{a.resname}{a.resid}:{a.name}" for a in sasa.ag)
	else:
		columns.extend(sasa.get_molecule_sasa())
		labels.extend(f"molecule {i}" for i in range(sasa.n_molecules))
	if args.total:
		columns.append(sasa.get_sasa())
		labels.append('total')
	n_frames = len(sasa.frames)
	return np.array(columns).reshape(len(columns), n_frames), labels


def _write_xvg(path: Path, times: np.ndarray, columns: np.ndarray, labels: list[str], command: str) -> None:
	with open(path, 'w') as out:
		out.write(f"# created by fastsasa {__version__}\n# command: {command}\n")
		out.write('@    title "Solvent accessible surface"\n')
		out.write('@    xaxis  label "Time (ps)"\n')
		out.write('@    yaxis  label "Area (nm\\S2\\N)"\n')
		out.write('@TYPE xy\n')
		for i, label in enumerate(labels):
			out.write(f'@ s{i} legend "{label}"\n')
		np.savetxt(out, np.column_stack([times, columns.T]), fmt='%12.4f')


def main(argv: Sequence[str] | None = None) -> int:
	args = parser().parse_args(argv)
	if args.per_atom and args.group:
		parser().error('per-atom and group cannot be combined')
	if args.threads is not None:
		import numba
		numba.set_num_threads(args.threads)

	import MDAnalysis as mda
	universe = mda.Universe(args.topology, *args.trajectory)
	if not hasattr(universe, 'trajectory'):
		print(f"fastsasa: error: {args.topology} has no coordinates; give a trajectory with -f", file=sys.stderr)
		return 1
	ag = universe.select_atoms(args.select)
	if not ag.n_atoms:
		print(f"fastsasa: error: the selection {args.select!r} contains no atoms", file=sys.stderr)
		return 1
	for group in args.group: # check the groups before the (long) calculation
		if not ag.select_atoms(group).n_atoms:
			print(f"fastsasa: error: the group {group!r} contains no atoms of the selection", file=sys.stderr)
			return 1

	sasa = SASA(ag, cg=args.cg, probe=args.probe, ndots=args.ndots, radii=dict(args.radius) or None,
	            max_frame_atoms=args.max_frame_atoms)
	if not args.quiet:
		print(f"{ag.n_atoms} atoms in {sasa.n_molecules} molecule(s), "
		      f"{len(universe.trajectory[args.start:args.stop:args.step])} frame(s)", file=sys.stderr)
	sasa.run(start=args.start, stop=args.stop, step=args.step, verbose=not args.quiet)

	columns, labels = _columns(sasa, args)
	output = Path(args.output)
	if output.suffix == '.npz':
		np.savez(output, times=sasa.times, frames=sasa.frames, sasa=columns, labels=np.array(labels))
	else:
		command = 'fastsasa ' + shlex.join(sys.argv[1:] if argv is None else argv)
		_write_xvg(output, sasa.times, columns, labels, command)
	if not args.quiet:
		print(f"wrote {len(labels)} column(s) x {len(sasa.frames)} frame(s) to {output}", file=sys.stderr)
	return 0


if __name__ == '__main__':
	sys.exit(main())
