"""Run one tiny end-to-end training check using the current Python environment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["snot", "sdon"], required=True)
    parser.add_argument("--case", choices=["lug3d", "dogbone"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2",
               TF_CPP_MIN_LOG_LEVEL="2", CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg")

    def run(command):
        subprocess.run([sys.executable, *map(str, command)], cwd=root, env=env, check=True)

    with tempfile.TemporaryDirectory(prefix="t-snot-smoke-") as temporary:
        data = Path(temporary) / "data"
        output = Path(temporary) / "output"
        run(["scripts/make_smoke_data.py", "--output-dir", data])
        command = ["-m", f"{args.model}.{args.case}.train", "--data-dir", data / args.case,
                   "--output-dir", output, "--num-samples", "10", "--batch-size", "2", "--cpu"]
        if args.case == "dogbone":
            command += ["--train-index", data / "dogbone/TrainIndex.npy"]
        if args.model == "snot":
            command += ["--epochs", "1", "--eval-batch-size", "2"]
        else:
            command += ["--iterations", "1"]
        run(command)
        run(["scripts/check_smoke_outputs.py", "--output-dir", output,
             "--model", args.model, "--case", args.case])


if __name__ == "__main__":
    main()
