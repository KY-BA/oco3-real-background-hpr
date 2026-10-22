"""Launch the archived paper training configuration without changing its science.

Place this file in scripts/ in the release repository. The default is a dry run.
Python 3.8 compatible. No training module is imported by this wrapper.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


PAPER_SEEDS = (42, 52, 62, 72, 82)
MODELS = ("pure_cnn", "mixed_cnn", "hpr")


def hydra_string(value):
    # Hydra receives this as one argv item; shell interpolation is never used.
    return json.dumps(str(value), ensure_ascii=False)


def read_input_paths(config_path, data_dir):
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "OmegaConf is required even for dry-run validation. "
            "Install the repository training requirements first."
        ) from exc
    cfg = OmegaConf.load(str(config_path))
    OmegaConf.update(cfg, "data.path.directory", data_dir.as_posix())
    names = (
        "path_train_ds", "path_valid_ds", "path_valid_2_ds", "path_extra_valid_ds"
    )
    paths = []
    for name in names:
        value = OmegaConf.select(cfg, "data.init." + name)
        if value is None:
            raise ValueError("Missing input path in archived config: data.init." + name)
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            raise ValueError("Input did not resolve to an absolute path: " + str(path))
        paths.append((name, path.resolve()))
    return paths


def build_command(python_exe, repo_root, model, config_dir, data_dir, output_dir, seed):
    run_name = "paper_{}_s{}".format(model, seed)
    return [
        str(python_exe),
        str(repo_root / "main.py"),
        "--config-path", str(config_dir),
        "--config-name", model,
        "seed={}".format(seed),
        "data.path.directory=" + hydra_string(data_dir.as_posix()),
        "dir_res=" + hydra_string(output_dir.as_posix()),
        "exp_name=" + run_name,
        "hydra.run.dir=${dir_res}/${exp_name}",
        "hydra.job.chdir=true",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Mixed: folder containing the four H5 files. Pure: parent of 2km_forOco3_32/.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--execute", action="store_true", help="Actually train; otherwise validate and print only.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1],
        help="Normally inferred from this script's location in scripts/.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config_dir = repo_root / "configs" / "training"
    config_path = config_dir / (args.model + ".yaml")
    if not (repo_root / "main.py").is_file():
        parser.error("Repository main.py is missing: " + str(repo_root))
    if not config_path.is_file():
        parser.error("Archived paper configuration is missing: " + str(config_path))
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide a nonempty seed list without duplicates.")
    if any(seed not in PAPER_SEEDS for seed in args.seeds):
        parser.error("This wrapper accepts only paper seeds: " + str(PAPER_SEEDS))

    try:
        input_paths = read_input_paths(config_path, data_dir)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    missing = sorted(set(str(path) for _, path in input_paths if not path.is_file()))
    if missing:
        parser.error("Required external input files are missing:\n" + "\n".join(missing))

    # Check every planned run before starting any of them. No overwrite/resume.
    run_dirs = [output_dir / "paper_{}_s{}".format(args.model, seed) for seed in args.seeds]
    conflicts = [str(path) for path in run_dirs if path.exists()]
    if conflicts:
        parser.error("Refusing to overwrite existing runs:\n" + "\n".join(conflicts))

    print("Mode: " + ("EXECUTE" if args.execute else "DRY RUN (no files written)"))
    print("Archived configuration: " + str(config_path))
    for name, path in input_paths:
        print("{}: {}".format(name, path))
    if tuple(args.seeds) != PAPER_SEEDS:
        print("Subset run: a paper ensemble requires all five paper seeds.")

    for seed, run_dir in zip(args.seeds, run_dirs):
        command = build_command(
            args.python_exe, repo_root, args.model, config_dir, data_dir, output_dir, seed
        )
        print("\nSeed {} -> {}".format(seed, run_dir))
        print(json.dumps(command, ensure_ascii=False, indent=2))
        if args.execute:
            # Recheck immediately before launch; existing results remain protected.
            if run_dir.exists():
                raise FileExistsError("Run appeared after validation: " + str(run_dir))
            subprocess.run(command, cwd=str(repo_root), check=True, shell=False)
    if not args.execute:
        print("\nAdd --execute only after checking these inputs and destinations.")


if __name__ == "__main__":
    main()
