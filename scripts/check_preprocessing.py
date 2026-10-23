"""Check the real entry points for training-target independence from held-out values."""

import argparse
from contextlib import ExitStack, redirect_stdout
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg", DDE_BACKEND="tensorflow")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
             "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(name, "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
DOG_TRAIN = np.array([0, 1, 3, 4, 5, 6, 8, 9])


class PreprocessingCaptured(Exception):
    pass


def write_data(folder, case, stress, peeq):
    folder.mkdir()
    dimensions = 3 if case == "lug3d" else 2
    coords = np.arange(1, 6 * dimensions + 1).reshape(6, dimensions) / 1000.0
    loads = np.arange(1, 11)[:, None] * np.linspace(0, 1, 101)[None, :] / 10.0
    if case == "lug3d":
        for name, values in (("xyz_coords", coords), ("data_amp_5000", loads),
                             ("ystress_all", stress * 1e6), ("peeq_all", peeq)):
            np.save(folder / (name + ".npy"), values)
    else:
        np.save(folder / "Coords.npy", coords)
        np.save(folder / "TrainIndex.npy", DOG_TRAIN)
        for name, values in (("Amp", loads), ("Stress", stress), ("PEEQ", peeq)):
            np.savez_compressed(folder / (name + ".npz"), a=values)


def capture_run(model, case, data, output, complete_run=False):
    """Run all production preprocessing; intercept the call that would train weights."""
    module = importlib.import_module(f"{model}.{case}.train")
    argv = [module.__file__, "--data-dir", str(data), "--output-dir", str(output),
            "--num-samples", "10", "--batch-size", "2", "--cpu"]
    if case == "dogbone":
        argv += ["--train-index", str(data / "TrainIndex.npy")]
    if model == "snot":
        argv += ["--epochs", "1", "--eval-batch-size", "2"]
    else:
        argv += ["--iterations", "1"]
    captures = []

    def capture(train, test):
        captures.append({"train": np.array(train, copy=True), "test": np.array(test, copy=True)})
        assert len(captures) == 1, "Training was called more than once"
        if not complete_run:
            raise PreprocessingCaptured

    def capture_snot(self, train_loader, *, val_loader, **kwargs):
        capture(train_loader.dataset.tensors[1].numpy(), val_loader.dataset.tensors[1].numpy())

    def capture_sdon(self, *args, **kwargs):
        capture(self.data.train_y, self.data.test_y)
        return [], None

    old_cwd = Path.cwd()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if model == "snot":
                from snot.trainer.torch_trainer import TorchTrainer
                stack.enter_context(patch.object(TorchTrainer, "fit", capture_snot))
            else:
                import deepxde as dde
                stack.enter_context(patch.object(dde.Model, "train", capture_sdon))
                if complete_run:
                    # Even deliberately poor test predictions must not start another training run.
                    stack.enter_context(patch.object(dde.Model, "predict",
                                                    lambda self, *a, **k: np.full_like(self.data.test_y, -10.0)))
            try:
                module.main()
            except PreprocessingCaptured:
                assert not complete_run
            else:
                assert complete_run, "Training entry point did not reach the capture hook"
    finally:
        os.chdir(old_cwd)
    for trial, values in enumerate(captures, 1):
        suffix = f"_trial_{trial}" if (model, case) == ("sdon", "lug3d") else ""
        values["indices"] = np.load(output / f"train_indices{suffix}.npy")
        test_indices = np.load(output / f"test_indices{suffix}.npy")
        np.testing.assert_array_equal(test_indices, np.setdiff1d(np.arange(10), values["indices"]))
    return captures


def check_training_scale(model, targets):
    assert np.isfinite(targets).all()
    if model == "snot":
        # Accumulate in float64 when checking float32 normalized fields.
        values = targets.astype(np.float64)
        np.testing.assert_allclose(values.mean(axis=(0, 1, 2)), 0, atol=2e-6)
        np.testing.assert_allclose(values.std(axis=(0, 1, 2)), 1, atol=2e-6)
    else:
        np.testing.assert_allclose(targets.min(axis=(0, 1)), 0, atol=1e-12)
        maxima = targets.max(axis=(0, 1))
        assert np.all(np.isclose(maxima, 0) | np.isclose(maxima, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["snot", "sdon"], required=True)
    parser.add_argument("--case", choices=["lug3d", "dogbone"], required=True)
    args = parser.parse_args()
    expected_train = (np.random.RandomState(2024).choice(10, 8, replace=False)
                      if args.case == "lug3d" else DOG_TRAIN)
    held_out = np.setdiff1d(np.arange(10), expected_train)
    frames = 101 if args.case == "lug3d" else 40
    field = (np.arange(1, 11)[:, None, None] * np.linspace(0, 1, frames)[None, :, None]
             * np.arange(1, 7)[None, None, :] / 6.0)
    stress, peeq = 10 * field, 0.001 * field
    stress[:, :, 0], peeq[:, :, 0] = 20, 0  # Include constant per-node features.
    with tempfile.TemporaryDirectory(prefix="t-snot-preprocessing-") as temporary:
        root = Path(temporary)
        original, changed, constant = (root / name for name in ("original", "changed", "constant"))
        write_data(original, args.case, stress, peeq)
        changed_s, changed_p = stress.copy(), peeq.copy()
        changed_s[held_out] += 7.0
        changed_p[held_out] += 0.2  # Put the global PEEQ maximum in the held-out set.
        write_data(changed, args.case, changed_s, changed_p)
        before = capture_run(args.model, args.case, original, root / "before")[0]
        after = capture_run(args.model, args.case, changed, root / "after")[0]
        for values in (before, after):
            np.testing.assert_array_equal(values["indices"], expected_train)
        np.testing.assert_array_equal(before["train"], after["train"],
                                      err_msg="Held-out targets changed the processed training targets")
        assert not np.array_equal(before["test"], after["test"])
        check_training_scale(args.model, after["train"])
        if args.model == "sdon":
            assert np.any(after["test"] > 1), "Held-out values must use the training scaler without refitting"

        constant_s, constant_p = stress.copy(), peeq.copy()
        constant_s[expected_train], constant_p[expected_train] = 77.0, 0.0
        write_data(constant, args.case, constant_s, constant_p)
        values = capture_run(args.model, args.case, constant, root / "zero_variance")[0]
        np.testing.assert_allclose(values["train"], 0, atol=1e-12)
        assert np.isfinite(values["test"]).all()

        if (args.model, args.case) == ("sdon", "lug3d"):
            single_data, single_output = root / "single_data", root / "single_output"
            write_data(single_data, args.case, 10 * field, 0.001 * field)
            captures = capture_run(args.model, args.case, single_data, single_output, complete_run=True)
            assert len(captures) == 1
            np.testing.assert_array_equal(captures[0]["indices"], expected_train)
            check_training_scale(args.model, captures[0]["train"])
            assert [p.name for p in single_output.glob('model_1br_*.npz')] == ['model_1br_1.npz']
            with np.load(single_output / 'model_1br_1.npz') as result:
                error = np.linalg.norm(result['b'] - result['d'], axis=-1) / (np.linalg.norm(result['b'], axis=-1) + 0.0001)
                assert error.mean() > 0.1, "The test must exceed the removed retry threshold"
    print(f"PASS {args.model}/{args.case}: held-out independence, split preservation, training scaling, constant targets"
          + (", exactly one training run despite high test error" if (args.model, args.case) == ("sdon", "lug3d") else ""))


if __name__ == "__main__":
    main()
