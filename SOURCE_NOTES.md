# Source provenance and packaging changes

The four entry points are based on these research scripts:

| Public entry point | Original filename |
|---|---|
| `snot/lug3d/train.py` | `s_not_lug3d_nsteps_gru.py` |
| `snot/dogbone/train.py` | `s_not_dog_bone_nsteps.py` |
| `sdon/lug3d/train.py` | `test_3d_steps_lug_MAE.py` |
| `sdon/dogbone/train.py` | `DeepXDE_GRU_vector.py` |

The sources are the full-history variants. The last-frame-only S-DON copies
and the 101-output-frame predecessor of the lug script are not used here.
Original file hashes are recorded in `source_manifest.json`.

## Changes for this distribution

- Added command-line paths, training-length and batch-size controls, a CPU
  switch, and `main()` guards. Input paths are resolved before outputs are
  redirected into a dedicated directory. Existing nonempty output directories
  are rejected to preserve prior runs.
- Enabled the three previously commented S-NOT lug training/logging lines.
  The default remains the source's 2,000 epochs. A separate historical
  4,000-epoch run is not used to change that default.
- Shared the identical S-NOT attention, position-encoding, and training
  modules between the two cases. Their file contents are unchanged.
- Preserved the original dog-bone training-index array and added validation
  for custom split files. Each run saves the train/test indices it uses.
- Corrected output preprocessing to fit only on training cases after splitting.
  This changes the source protocol: S-NOT uses training means/standard deviations,
  S-DON uses training per-node extrema, and dog-bone's PEEQ cap uses 60% of the
  training maximum. Both splits use the same fitted transform and cap.
  S-NOT uses a scale of 1 when
  a training field has zero variance. Original data and first-run split rules
  are preserved.
- Removed the S-DON lug test-error-driven retry loop and `--max-trials` option.
  It trains exactly once on the seeded split for the requested number of
  iterations. Existing first-run output filenames, including `model_1br_1.npz`
  and `train_indices_trial_1.npy`, are retained for compatibility.
- Added explicit PyTorch/NumPy/Python seeds for S-NOT runs. This makes the
  initialization configurable; it does not promise identical results across
  devices or reproduce an unspecified historical initialization.
- Selected the TensorFlow backend explicitly for S-DON and pinned a Keras 2
  environment. The source's model architectures, loss functions, optimizers,
  schedules, and output transforms are retained.
- Derived the S-DON lug node count from the coordinates; the research data
  still produce the original count of 29,852 nodes.
- Corrected S-NOT dog-bone's post-training stress error indexing from
  `[:, :, 0]` (node selection) to `[..., 0]` (stress-channel selection).
  This changes the printed error calculation, not the training loss.
- Corrected S-DON lug's post-training relative absolute error calculation to
  retain the time-axis mean dimension when broadcasting. This does not
  change training or predictions.
- Removed dormant notebook-style code blocks, formatted the entry points,
  and removed cluster-specific paths from the distributed source.
- Added synthetic execution checks and documentation. No FEA datasets,
  pretrained weights, or large research result archives are bundled.

## Limits

The original pooled preprocessing was corrected as described above. Its effect
on full-data accuracy has not been measured by retraining. S-NOT's use of test
data for checkpoint selection and learning-rate adjustment remains.
S-DON dog-bone error clipping also remains. S-DON lug's retry rule has been removed.
The original scripts here do not provide the complete five-fold evaluation
workflow described in the paper.

Synthetic checks establish that the executable paths train, save a checkpoint,
and produce finite predictions with the expected axes. They do not establish
agreement with the paper's aggregate accuracy or runtime measurements.
