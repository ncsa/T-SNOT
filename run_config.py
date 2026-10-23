"""Command-line and output-directory setup shared by the four training scripts."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_run_args(family, case):
    parser = argparse.ArgumentParser(description=f"Train {family.upper()} on {case}.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the input arrays listed in data/README.md.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Empty directory for checkpoints, predictions and logs.")
    parser.add_argument("--num-samples", type=positive_int,
                        default=1000 if case == "lug3d" else 4000)
    parser.add_argument("--seed", type=int,
                        default=123 if (family, case) == ("sdon", "dogbone") else 2024)
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA for this run.")
    parser.add_argument("--batch-size", type=positive_int,
                        default=(8 if case == "lug3d" else 32) if family == "snot" else 64)
    if family == "snot":
        parser.add_argument("--epochs", type=positive_int,
                            default=2000 if case == "lug3d" else 1500)
        parser.add_argument("--eval-batch-size", type=positive_int,
                            default=32 if case == "lug3d" else 64)
    else:
        parser.add_argument("--iterations", type=positive_int,
                            default=100000 if case == "lug3d" else 300000)
    if case == "dogbone":
        parser.add_argument("--train-index", type=Path,
                            default=ROOT / "data/dogbone/TrainIndex.npy",
                            help="Shared S-NOT/S-DON training indices for the selected samples.")
    return parser.parse_args()


def prepare_run(args, family, case):
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.num_samples < 2:
        raise ValueError("At least two samples are needed for training and testing.")
    names = (["peeq_all.npy", "ystress_all.npy", "data_amp_5000.npy", "xyz_coords.npy"]
             if case == "lug3d" else ["PEEQ.npz", "Stress.npz", "Amp.npz", "Coords.npy"])
    missing = [str(args.data_dir / name) for name in names
               if not (args.data_dir / name).is_file()]
    if case == "dogbone":
        args.train_index = args.train_index.expanduser().resolve()
        if not args.train_index.is_file():
            missing.append(str(args.train_index))
    if missing:
        raise FileNotFoundError("Required data files are missing:\n" + "\n".join(missing)
                                + "\nSee data/README.md.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Choose a new directory.")
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(model=family, case=case)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    os.chdir(args.output_dir)
    if family == "sdon":
        Path("Models" if case == "lug3d" else "mdls").mkdir(exist_ok=True)


def load_train_index(path, sample_count):
    import numpy as np

    indices = np.load(path, allow_pickle=False)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or not 0 < len(indices) < sample_count
            or np.any(indices < 0) or np.any(indices >= sample_count)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError(f"{path} must contain unique training indices in [0, {sample_count}). "
                         "Supply a matching --train-index when using a smaller dataset.")
    return indices
