"""Check that a tiny training run produced finite predictions and a checkpoint."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["snot", "sdon"], required=True)
    parser.add_argument("--case", choices=["lug3d", "dogbone"], required=True)
    args = parser.parse_args()
    if args.model == "snot":
        name = "predictions.npz"
        checkpoint = list((args.output_dir / "checkpoints").glob("*.ckpt"))
        assert (args.output_dir / "checkpoints/logs.json").is_file()
    elif args.case == "lug3d":
        name = "model_1br_1.npz"
        checkpoint = list((args.output_dir / "Models").glob("*.ckpt.index"))
        assert len(checkpoint) == 1, "S-DON lug must save one training run"
        assert len(list(args.output_dir.glob("model_1br_*.npz"))) == 1
        assert len(list(args.output_dir.glob("train_indices_trial_*.npy"))) == 1
        np.testing.assert_array_equal(np.load(args.output_dir / "train_indices_trial_1.npy"),
                                      np.random.RandomState(2024).choice(10, 8, replace=False))
    else:
        name = "TestData_minmax_HD32_vec2_relu.npz"
        checkpoint = list((args.output_dir / "mdls").glob("*.ckpt.index"))
    assert checkpoint, f"No saved checkpoint in {args.output_dir}"
    expected = ((2, 6, 100) if args.model == "snot" else (2, 100, 6)) if args.case == "lug3d" else (2, 40, 6)
    with np.load(args.output_dir / name, allow_pickle=False) as results:
        for key in ["a", "b", "c", "d"]:
            values = results[key]
            assert values.shape == expected, (key, values.shape, expected)
            assert np.isfinite(values).all(), f"Non-finite values in {key}"
    assert (args.output_dir / "run_config.json").is_file()
    print(f"PASS {args.model}/{args.case}: checkpoint and finite predictions {expected}")


if __name__ == "__main__":
    main()
