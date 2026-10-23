# T-SNOT

Training code for **S-NOT** (Sequential Neural Operator Transformer) and **S-DON**
(Sequential DeepONet) on the **3D lug** and **dog-bone** plasticity benchmarks in:

> Jaewan Park, Kazuma Kobayashi, Qibang Liu, Amar Alem Koric, Diab W. Abueidda,
> Syed Bahauddin Alam, and Seid Koric. **Sequential deep operator neural networks
> for plastic and thermo-viscoplastic transient material behavior.**
> *International Journal of Plasticity*, 201 (2026), 104689.
> [https://doi.org/10.1016/j.ijplas.2026.104689](https://doi.org/10.1016/j.ijplas.2026.104689)

The repository address named in the paper is [ncsa/T-SNOT](https://github.com/ncsa/T-SNOT).
This package contains four full-history training experiments. The paper's steel
solidification and perturbation studies are outside this package's scope.

## Experiments

| Model | Case | Entry point | Default training | Samples | Batch size |
|---|---|---|---|---:|---:|
| S-NOT | 3D lug | `python -m snot.lug3d.train` | 2,000 epochs; AdamW; MAE | 1,000 | 8 |
| S-NOT | Dog-bone | `python -m snot.dogbone.train` | 1,500 epochs; Adam; MAE | 4,000 | 32 |
| S-DON | 3D lug | `python -m sdon.lug3d.train` | 100,000 iterations; Adam; MAE | 1,000 | 64 |
| S-DON | Dog-bone | `python -m sdon.dogbone.train` | 300,000 iterations; Adam; MSE | 4,000 | 64 |

Both models predict stress and equivalent plastic strain (PEEQ) at 100 output
steps for the lug and 40 output steps for the dog-bone. All four accept load
histories sampled at 101 input points. S-NOT uses PyTorch; S-DON uses DeepXDE
with the TensorFlow 2 backend.

## Installation

Use **Python 3.10** and separate environments for the two frameworks. Clone the
repository and enter its root directory:

```bash
git clone https://github.com/ncsa/T-SNOT.git
cd T-SNOT
```

Run all remaining commands from `T-SNOT/`. Create the S-NOT environment:

```bash
python3.10 -m venv .venv-snot
source .venv-snot/bin/activate
python -m pip install -r requirements-snot.txt
```

For S-DON, create a separate environment:

```bash
python3.10 -m venv .venv-sdon
source .venv-sdon/bin/activate
python -m pip install -r requirements-sdon.txt
```

The S-DON environment pins TensorFlow/Keras 2.14 and TensorFlow Probability
0.22.1 to retain the original Keras 2 API. The scripts select `DDE_BACKEND=tensorflow`
before importing DeepXDE. See the [DeepXDE installation guide](https://deepxde.readthedocs.io/en/latest/user/installation.html)
and [TensorFlow Probability 0.22.1 release notes](https://github.com/tensorflow/probability/releases/tag/v0.22.1).

Install a PyTorch or TensorFlow GPU build appropriate to your hardware for full
training. `--cpu` is available for small checks. Full-resolution training has
substantial memory requirements; the synthetic checks below do not estimate
full-dataset memory or accuracy.

## Data

Download the input histories, coordinates, and reference output fields from
**[Hugging Face: jaewan-wod33/T-SNOT-data](https://huggingface.co/datasets/jaewan-wod33/T-SNOT-data)**.
The dataset contains 1,000 lug cases and 4,000 dog-bone cases, including both
training and test cases. The download is approximately **50.7 GB**. Both models
use the same data for each case.

1. Sign in to Hugging Face and open the dataset link above.
2. Select **Request access**, fill in your affiliation and intended use, and
   submit the form. The dataset owner reviews requests manually.
3. After approval, run the following commands from `T-SNOT/`, with either of
   the Python environments above activated. Follow the login prompts using
   the **same Hugging Face account that received access**. If prompted for a
   token, create one with read access in your
   [Hugging Face token settings](https://huggingface.co/settings/tokens).

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf download jaewan-wod33/T-SNOT-data --repo-type dataset --local-dir ./benchmark-data
```

The download command creates the following layout automatically. If you have
already downloaded the files, place the complete `lug3d/` and `dogbone/` folders
inside `T-SNOT/benchmark-data/`, keeping their filenames unchanged:

```text
T-SNOT/
├── README.md
├── snot/
├── sdon/
└── benchmark-data/
    ├── lug3d/
    │   ├── data_amp_5000.npy
    │   ├── xyz_coords.npy
    │   ├── ystress_all.npy
    │   ├── peeq_all.npy
    │   └── node_labels_sorted.npy
    └── dogbone/
        ├── Amp.npz
        ├── Coords.npy
        ├── Stress.npz
        ├── PEEQ.npz
        └── TrainIndex.npy
```

Additional dataset metadata files are also downloaded into `benchmark-data/`.
Keep `.npz` files in their downloaded form; the training scripts load them
directly. `node_labels_sorted.npy` is optional for training.

The four commands below use this layout without editing any Python files.
For data stored elsewhere, replace `--data-dir` with the corresponding case
directory, such as `/path/to/benchmark-data/lug3d`. For dog-bone, also point
`--train-index` to `/path/to/benchmark-data/dogbone/TrainIndex.npy`.

The downloaded `TrainIndex.npy` contains the original 3,200 training indices
and is identical to the copy bundled at `data/dogbone/TrainIndex.npy`. Both
dog-bone models use the remaining 800 cases for testing. See
[data/README.md](data/README.md) for array shapes, units, and preprocessing,
and the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
for download options.

## Training

Provide the input directory and a new, empty output directory. Relative input
paths are resolved before any outputs are written. Checkpoints, prediction
arrays, split indices, and `run_config.json` are saved under `--output-dir`.
Run your chosen experiment from `T-SNOT/` after downloading the data above.

```bash
# S-NOT environment
source .venv-snot/bin/activate
python -m snot.lug3d.train --data-dir benchmark-data/lug3d --output-dir outputs/snot-lug3d
python -m snot.dogbone.train --data-dir benchmark-data/dogbone --train-index benchmark-data/dogbone/TrainIndex.npy --output-dir outputs/snot-dogbone

# S-DON environment
source .venv-sdon/bin/activate
python -m sdon.lug3d.train --data-dir benchmark-data/lug3d --output-dir outputs/sdon-lug3d
python -m sdon.dogbone.train --data-dir benchmark-data/dogbone --train-index benchmark-data/dogbone/TrainIndex.npy --output-dir outputs/sdon-dogbone
```

Results are written to the corresponding `T-SNOT/outputs/` subdirectory.
Choose a new `--output-dir` for each run; an existing nonempty output directory
is rejected.

Use `--help` on any entry point to see its controls. `--epochs` applies to S-NOT;
`--iterations` applies to S-DON. Both accept `--batch-size`, `--num-samples`,
`--seed`, and `--cpu`. S-NOT also accepts `--eval-batch-size`.

S-DON lug trains once on one seeded 80/20 split, for the requested number of
`--iterations`. Test error is reported after training and does not trigger
another training run or change the split.

## Small execution checks

These commands generate synthetic data in a temporary directory, run one epoch
or iteration on the CPU, and verify checkpoint creation, prediction shapes,
and finite values. They do not test agreement with the paper's reported errors.

```bash
# S-NOT environment
python scripts/smoke_test.py --model snot --case lug3d
python scripts/smoke_test.py --model snot --case dogbone

# S-DON environment
python scripts/smoke_test.py --model sdon --case lug3d
python scripts/smoke_test.py --model sdon --case dogbone
```

The same four checks are configured in `.github/workflows/smoke.yml`.
See [VALIDATION.md](VALIDATION.md) for the local validation results and scope.

## Outputs

| Experiment | Prediction archive | Keys `a`, `b`, `c`, `d` | Field array axes |
|---|---|---|---|
| S-NOT lug | `predictions.npz` | PEEQ true, stress true, PEEQ predicted, stress predicted | case, node, time |
| S-NOT dog-bone | `predictions.npz` | PEEQ true, stress true, PEEQ predicted, stress predicted | case, time, node |
| S-DON lug | `model_1br_1.npz` | PEEQ true, stress true, PEEQ predicted, stress predicted | case, time, node |
| S-DON dog-bone | `TestData_minmax_HD32_vec2_relu.npz` | stress true, stress predicted, PEEQ true, PEEQ predicted | case, time, node |

Stress outputs are in MPa and PEEQ is dimensionless. S-NOT archives also include
coordinates under `e`. S-DON lug uses `e` for test indices; S-DON dog-bone uses
`e` for input histories and `f` for coordinates. Checkpoints reside in
`checkpoints/` (S-NOT), `Models/` (S-DON lug), or `mdls/` (S-DON dog-bone).

## Source provenance and reproduction scope

[source_manifest.json](source_manifest.json) records the original source
filenames and SHA-256 hashes. [SOURCE_NOTES.md](SOURCE_NOTES.md) describes the
packaging changes, including enabling lug training and correcting two
postprocessing indexing/broadcasting issues. Shared S-NOT model modules and
the trainer were copied verbatim, including their author notices.

The models, losses, and default training schedules follow the supplied research
scripts. Output preprocessing has been corrected: split cases first, then fit
S-NOT's means/standard deviations or S-DON's per-node MinMax scalers using only
training targets. Dog-bone's PEEQ cap is 0.6 times the training-set maximum;
the fixed stress cap remains 260 MPa. The fitted transforms and clipping bounds
are applied to both train and test cases. S-DON lug uses one fixed split and
fits its scalers once. Constant S-NOT training fields use a scale of 1.
These changes affect the training targets and do not reproduce the original
pooled-preprocessing protocol or establish agreement with its reported results.

Other evaluation limitations remain: S-NOT uses its test loader for checkpoint
selection and learning-rate adjustment. S-DON dog-bone retains clipping of saved
and printed relative errors at 1.0. Correcting preprocessing and removing the
S-DON lug retry rule does not resolve those separate limitations.

These entry points provide the source scripts' 80/20 split workflow, including
the shared dog-bone split. They do not supply the complete five-fold evaluation
pipeline described in the paper. Full-data retraining is required to assess
numerical agreement; a successful synthetic run alone cannot establish it.

## Citation

GitHub's citation metadata is provided in [CITATION.cff](CITATION.cff). When using
these experiments, cite the paper:

```bibtex
@article{Park2026Sequential,
  title = {Sequential deep operator neural networks for plastic and thermo-viscoplastic transient material behavior},
  author = {Park, Jaewan and Kobayashi, Kazuma and Liu, Qibang and Koric, Amar Alem and Abueidda, Diab W. and Alam, Syed Bahauddin and Koric, Seid},
  journal = {International Journal of Plasticity},
  volume = {201},
  pages = {104689},
  year = {2026},
  doi = {10.1016/j.ijplas.2026.104689}
}
```

## License

Project-authored code is distributed under the [MIT License](LICENSE).
Third-party-derived portions retain their upstream terms and attribution;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
