"""Periodic-interface umbrella sampling and per-model WHAM reconstruction."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def solve_wham(samples, centers, k=5.0, temperature=300.0, bins=120, edges=None):
    """Return bin centers and PMF (eV); unsampled bins remain NaN."""
    samples = [np.asarray(x, dtype=float) for x in samples]
    if (not samples or len(samples) != len(centers) or k <= 0 or temperature <= 0
            or bins < 2 or any(x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all()
                             for x in samples) or not np.isfinite(centers).all()):
        raise ValueError("WHAM needs finite, nonempty windows and positive settings.")
    values = np.concatenate(samples)
    if edges is None:
        edges = np.linspace(values.min() - 0.01, values.max() + 0.01, bins + 1)
    edges = np.asarray(edges)
    if (len(edges) < 3 or not np.all(np.diff(edges) > 0)
            or not np.allclose(np.diff(edges), np.diff(edges)[0])
            or values.min() < edges[0] or values.max() > edges[-1]):
        raise ValueError("Use uniform histogram edges covering all samples.")
    mids = (edges[1:] + edges[:-1]) / 2
    hist = np.array([np.histogram(x, edges)[0] for x in samples])
    overlap = (hist > 0).astype(int) @ (hist > 0).astype(int).T > 0
    connected = {0}
    while True:
        expanded = connected | set(np.where(overlap[list(connected)].any(axis=0))[0])
        if expanded == connected:
            break
        connected = expanded
    if len(connected) != len(samples):
        raise ValueError("Window histograms are disconnected; extend sampling/overlap.")
    supported = hist.sum(axis=0) > 0
    beta = 1 / (8.617333262e-5 * temperature)
    bias = 0.5 * k * (mids[supported][None, :] - np.asarray(centers)[:, None]) ** 2
    log_counts = np.log(hist[:, supported].sum(axis=0))
    log_n = np.log([len(x) for x in samples])[:, None]
    offsets = np.zeros(len(samples))
    for _ in range(20000):
        log_p = log_counts - logsumexp(log_n + beta * (offsets[:, None] - bias), axis=0)
        updated = -logsumexp(log_p[None, :] - beta * bias, axis=1) / beta
        updated -= updated[0]
        delta = np.max(np.abs(updated - offsets))
        offsets = updated
        if delta < 1e-9:
            break
    else:
        raise RuntimeError("WHAM did not converge.")
    log_p = log_counts - logsumexp(log_n + beta * (offsets[:, None] - bias), axis=0)
    pmf = np.full(len(mids), np.nan)
    pmf[supported] = -log_p / beta
    pmf[supported] -= np.nanmin(pmf)
    return mids, pmf


def distance_bias(i, j, center, k):
    from ase.calculators.calculator import Calculator, all_changes

    class Bias(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            vector = atoms.get_distance(i, j, mic=True, vector=True)
            distance = np.linalg.norm(vector)
            if distance < 1e-10:
                raise ValueError("The reaction-coordinate atoms coincide.")
            forces = np.zeros((len(atoms), 3))
            forces[i] = k * (distance - center) * vector / distance
            forces[j] = -forces[i]
            self.results = {"energy": 0.5 * k * (distance - center) ** 2, "forces": forces}

    return Bias()


def sample(args):
    from ase import units
    from ase.calculators.mixing import SumCalculator
    from ase.constraints import Hookean
    from ase.io import read
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mace.calculators import MACECalculator

    atoms = read(args.structure)
    i, j = args.atoms
    if i == j or min(i, j) < 0 or max(i, j) >= len(atoms):
        raise ValueError("--atoms must specify two distinct, valid zero-based indices.")
    if not atoms.pbc.all() or atoms.constraints:
        raise ValueError("Supply an unconstrained, fully periodic interface structure.")
    if (args.stop < args.start or min(args.start, args.step, args.k, args.temperature,
                                    args.dt, args.steps, args.save_every) <= 0
            or args.warmup < 0 or args.oxide_k < 0 or args.steps // args.save_every < 2):
        raise ValueError("Invalid window or dynamics settings.")
    centers = np.arange(args.start, args.stop + args.step * 1e-8, args.step)
    if len(set(f"{x:.6f}" for x in centers)) != len(centers):
        raise ValueError("Window spacing is too small for output filenames.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    for model_index, model in enumerate(args.models):
        calculator = MACECalculator(model_paths=model, device=args.device, default_dtype="float32")
        folder = out / f"model_{model_index + 1}"
        folder.mkdir()
        for window_index, center in enumerate(centers):
            current = atoms.copy()
            if args.oxide_k:
                current.set_constraint([Hookean(a.index, a.position.copy(), args.oxide_k, rt=0)
                                        for a in current if a.symbol in {"Hf", "O", "Sc", "Y"}])
            current.calc = SumCalculator([calculator, distance_bias(i, j, center, args.k)])
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, model_index, window_index]))
            MaxwellBoltzmannDistribution(current, temperature_K=args.temperature, rng=rng)
            dynamics = Langevin(current, args.dt * units.fs, temperature_K=args.temperature,
                                friction=0.02, fixcm=False, rng=rng)
            dynamics.run(args.warmup)
            rows = []
            for step in range(args.steps):
                dynamics.run(1)
                if (step + 1) % args.save_every == 0:
                    rows.append(((step + 1) * args.dt, current.get_distance(i, j, mic=True)))
            np.savetxt(folder / f"colvar_win_{center:.6f}.dat", rows,
                       header=f"time_fs CV_A center={center} k={args.k} T={args.temperature}")


def reconstruct(args):
    if not 0 <= args.discard < 1:
        raise ValueError("--discard must be in [0, 1).")
    groups = []
    for folder in args.inputs:
        files = sorted(Path(folder).glob("colvar_win_*.dat"))
        if not files:
            raise ValueError(f"No window files in {folder}")
        centers, samples = [], []
        for file in files:
            data = np.loadtxt(file, ndmin=2)
            if data.shape[1] != 2 or len(data) < 2 or not np.isfinite(data).all():
                raise ValueError(f"Invalid time/CV table: {file}")
            centers.append(float(file.stem.removeprefix("colvar_win_")))
            samples.append(data[int(len(data) * args.discard):, 1])
        groups.append((centers, samples))
    values = np.concatenate([x for _, samples in groups for x in samples])
    edges = np.linspace(values.min() - 0.01, values.max() + 0.01, args.bins + 1)
    profiles = []
    for centers, samples in groups:
        mids, profile = solve_wham(samples, centers, args.k, args.temperature, args.bins, edges)
        profiles.append(profile)
    profiles = np.asarray(profiles)
    shared = np.isfinite(profiles).all(axis=0)
    if not shared.any():
        raise ValueError("Models have no common sampled bins.")
    mean, std = np.full(len(mids), np.nan), np.full(len(mids), np.nan)
    mean[shared], std[shared] = profiles[:, shared].mean(axis=0), profiles[:, shared].std(axis=0)
    np.savetxt(args.out, np.column_stack([mids, profiles.T, mean, std]), delimiter=",",
               header="CV_A," + ",".join(f"model_{i + 1}_eV" for i in range(len(profiles)))
               + ",mean_eV,std_eV", comments="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("sample")
    run.add_argument("--structure", required=True)
    run.add_argument("--models", nargs="+", required=True)
    run.add_argument("--atoms", nargs=2, type=int, required=True, help="Zero-based Li and N/S indices")
    for name in ("start", "stop"):
        run.add_argument(f"--{name}", type=float, required=True)
    for name, value in (("step", 0.15), ("dt", 0.5), ("oxide-k", 0.0)):
        run.add_argument(f"--{name}", type=float, default=value)
    for name, value in (("steps", 20000), ("warmup", 500), ("save-every", 10), ("seed", 11)):
        run.add_argument(f"--{name}", type=int, default=value)
    run.add_argument("--device", default="cuda")
    wham = commands.add_parser("wham")
    wham.add_argument("--inputs", nargs="+", required=True)
    wham.add_argument("--discard", type=float, default=0.2)
    wham.add_argument("--bins", type=int, default=120)
    for command in (run, wham):
        command.add_argument("--k", type=float, default=5.0)
        command.add_argument("--temperature", type=float, default=300.0)
        command.add_argument("--out", required=True)
    args = parser.parse_args()
    (sample if args.command == "sample" else reconstruct)(args)


if __name__ == "__main__":
    main()
