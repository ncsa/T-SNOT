"""Create small synthetic arrays to test the training pipelines, not paper accuracy."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")
    rng = np.random.default_rng(2024)
    samples, nodes = 10, 6
    load_time = np.linspace(0, 1, 101)
    loads = rng.uniform(0.5, 1.5, (samples, 1)) * load_time[None, :]
    for case, frames, dimensions in [("lug3d", 101, 3), ("dogbone", 40, 2)]:
        folder = args.output_dir / case
        folder.mkdir(parents=True, exist_ok=True)
        coords = rng.uniform(0.001, 0.01, (nodes, dimensions))
        history = np.linspace(0, 1, frames)[None, :, None]
        spatial = rng.uniform(0.5, 1.5, (1, 1, nodes))
        field = loads[:, -1, None, None] * history * spatial
        stress = 100 * field
        peeq = 0.01 * field**2
        if case == "lug3d":
            np.save(folder / "xyz_coords.npy", coords)
            np.save(folder / "data_amp_5000.npy", loads)
            np.save(folder / "ystress_all.npy", stress * 1e6)
            np.save(folder / "peeq_all.npy", peeq)
        else:
            np.save(folder / "Coords.npy", coords)
            np.savez_compressed(folder / "Amp.npz", a=loads)
            np.savez_compressed(folder / "Stress.npz", a=stress)
            np.savez_compressed(folder / "PEEQ.npz", a=peeq)
            np.save(folder / "TrainIndex.npy", np.array([0, 1, 3, 4, 5, 6, 8, 9]))
    print(f"Created synthetic smoke data in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
