# Input data for the plasticity benchmarks

Place the arrays below in any directory and pass that directory with
`--data-dir`. Arrays are expected to be uncompressed `.npy` files where that
extension is listed. `.npz` files use the member key `a`.

The datasets are hosted separately from the code at
[jaewan-wod33/T-SNOT-data](https://huggingface.co/datasets/jaewan-wod33/T-SNOT-data).
Request access on the dataset page and wait for the owner's manual approval.
The release contains exactly the 1,000 lug and 4,000 dog-bone cases selected
by these scripts, including their
training and test splits. The smaller case pools preserve the original array
values, dtypes, order, and filenames and work with the existing default settings.

After access is granted, run these commands from the code repository root
(`T-SNOT/`) with a Python environment activated. Log in using the Hugging Face
account that received approval:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf download jaewan-wod33/T-SNOT-data --repo-type dataset --local-dir ./benchmark-data
```

Pass `--data-dir benchmark-data/lug3d` or
`--data-dir benchmark-data/dogbone`. For the dog-bone commands, pass
`--train-index benchmark-data/dogbone/TrainIndex.npy` as well. The supplied
dog-bone `TrainIndex.npy` is identical to the one bundled with this code. Cite DOI
`10.1016/j.ijplas.2026.104689` when using or requesting the benchmark data.
The corresponding author is Seid Koric (`koric@illinois.edu`).

See the main [README data instructions](../README.md#data) for the expected
folder layout and [training commands](../README.md#training) for all four
experiments. If the files are already downloaded, place the `lug3d/` and
`dogbone/` directories inside `T-SNOT/benchmark-data/`, or use their actual paths
with `--data-dir` and `--train-index`.

## 3D lug

| File | Release array shape | Meaning |
|---|---|---|
| `data_amp_5000.npy` | `(1000, 101)` | Dimensionless pressure-amplitude histories; pressure = amplitude × 100 MPa |
| `xyz_coords.npy` | `(29852, 3)` | Spatial coordinates in metres |
| `ystress_all.npy` | `(1000, 101, 29852)` | Von Mises stress fields in Pa, despite the historical filename |
| `peeq_all.npy` | `(1000, 101, 29852)` | Dimensionless equivalent plastic strain |

The upstream amplitude file contains 5,000 histories; the release preserves its
filename while selecting the first 1,000, which have matching output fields.
Both models omit the initial output frame, predicting 100 subsequent frames.
Stress is converted from Pa to MPa. S-NOT standardizes coordinates and fields;
S-DON multiplies coordinates by 1,000 and applies per-node MinMax scaling to
the fields. Output scaling is fitted on training cases only and applied to both
splits. Both models use the default 800/200 split with seed 2024. S-DON trains
once on that split, without test-error-driven retries. Test values may lie
outside the training MinMax range of `[0, 1]`; the scalers are not refitted on test.

## Dog-bone

| File | Release array shape | Meaning |
|---|---|---|
| `Amp.npz`, key `a` | `(4000, 101)` | Input x-displacement histories in mm |
| `Coords.npy` | `(3060, 2)` | Spatial coordinates in mm |
| `Stress.npz`, key `a` | `(4000, 40, 3060)` | Von Mises stress fields in MPa |
| `PEEQ.npz`, key `a` | `(4000, 40, 3060)` | Dimensionless equivalent plastic strain |

The upstream arrays contain 9,903 cases; this release contains their first
4,000 cases. The scripts predict all 40 output frames.
They cap stress at 260 MPa and PEEQ at 60% of the maximum over the selected
training cases. S-NOT uses standardization; S-DON uses MinMax scaling. Output
scaling is fitted after clipping, using training cases only. The same clipping
bounds and fitted transforms apply to test cases. This corrects the original
scripts' use of the full train-plus-test pool; the data files themselves are
unchanged.

`dogbone/TrainIndex.npy` is bundled because it is needed to use the same
3,200/800 split in both models. It contains only integer sample indices.
Its ordering is preserved. The test indices are the sorted complement within
the selected sample pool. `--train-index` can select another split file.

## Synthetic data

`scripts/make_smoke_data.py` produces ten synthetic cases and six spatial
nodes, retaining each problem's original input and output time counts. These
arrays are solely execution fixtures and are not finite-element simulations.

```bash
python scripts/make_smoke_data.py --output-dir smoke_data
```

For dog-bone checks, use `--train-index smoke_data/dogbone/TrainIndex.npy`
together with `--num-samples 10`. The original 3,200-index split is not valid
for this synthetic sample pool.
