# Validation of the public package

Validated locally on 2026-10-23 with Python 3.10.14 on Linux aarch64, using
CPU execution and the versions pinned in the requirements files:

- PyTorch 2.5.1
- TensorFlow 2.14.0, Keras 2.14.0, TensorFlow Probability 0.22.1
- DeepXDE 1.9.3
- NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.6.1, Matplotlib 3.9.2

## End-to-end results

All four documented `scripts/smoke_test.py` commands passed. Each generated ten
synthetic cases with six spatial nodes, trained on eight cases, and predicted
the remaining two. Original input and output time counts were retained.

| Experiment | Training performed | Verified prediction shape | Result |
|---|---|---|---|
| S-NOT lug | One epoch | `(2, 6, 100)` per field | Passed |
| S-NOT dog-bone | One epoch | `(2, 40, 6)` per field | Passed |
| S-DON lug | One iteration, single training run | `(2, 100, 6)` per field | Passed |
| S-DON dog-bone | One iteration | `(2, 40, 6)` per field | Passed |

Checks confirmed a saved checkpoint, finite values in all four prediction
archive field arrays, expected array axes, and a saved run configuration.
The S-NOT checks also confirmed saved training logs.
All four checks were rerun successfully after the training-only preprocessing
correction described in `SOURCE_NOTES.md`.

## Preprocessing regression checks

Each `scripts/check_preprocessing.py --model MODEL --case CASE` command runs
the actual training entry point and captures its processed targets at the
training API. It checks the data pipeline without updating model weights.
All four model/case combinations passed:

- Changing only held-out stress and PEEQ values leaves the processed training
  targets exactly unchanged. The held-out PEEQ maximum is deliberately made
  larger than every training value.
- The saved split preserves the original seeded lug rule or supplied dog-bone
  training indices, with test indices equal to the sorted complement.
- Training fields have the expected standardized or per-node MinMax scale.
  Held-out MinMax values can exceed 1 without refitting or clipping the scaler.
- Constant training fields produce finite transformed targets, including
  zero-variance S-NOT fields and constant S-DON node features.
- S-DON lug completes exactly one training call even when a stub prediction
  makes its test stress error exceed the former 0.1 retry threshold. The saved
  indices still match the first seeded split, and only one prediction archive
  is produced.

As a negative control, the same regression script was run against a separate
copy of the package before this correction. All four variants failed the
held-out-independence assertion, reproducing the original preprocessing leak.
These regression checks are also included in the GitHub workflow matrix.

After removing the S-DON lug retry loop, its end-to-end check was rerun without
any trial-limit option. It produced one checkpoint, one prediction archive,
and one split. A pre-removal copy that already had training-only preprocessing
fails the single-training-call assertion under the deliberately poor prediction.

## Additional checks

- All 20 Python files passed syntax parsing; the four training modules import
  without executing training. All four `--help` commands succeeded.
- Original source hashes remained unchanged. Shared S-NOT modules and the
  original dog-bone index file matched their sources byte for byte.
- All model and trainer class definitions matched the original abstract
  syntax trees after excluding docstring formatting.
- The original dog-bone split contains 3,200 valid training indices out of
  4,000 cases. A mismatched split for a smaller sample pool is rejected.
- Missing input files fail before a run directory is created.
- `CITATION.cff` passed the Citation File Format 1.2.0 schema via cffconvert.
- GitHub workflow YAML parsed with the intended four-experiment matrix;
  local documentation links resolve; distributed text contains no original
  cluster-specific absolute paths.
- `git diff --cached --check` passed.

## Scope

These are execution checks on synthetic arrays. Full-resolution FEA data
were not used for retraining, and the paper's accuracy and speed measurements
were not reproduced during packaging. GitHub-hosted workflow execution has
not been observed as part of this local validation.
