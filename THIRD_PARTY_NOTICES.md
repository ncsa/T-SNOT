# Third-party notices

The root MIT license applies to project-authored code. It does not replace
the license terms or notices associated with third-party-derived portions.

## S-NOT modules

`snot/modules/point_position_embedding.py` and
`snot/trainer/torch_trainer.py` retain Qibang Liu's original 2025 copyright
and author notices.

The position-encoding source identifies
[OpenAI Shap-E's encoding module](https://github.com/openai/shap-e/blob/main/shap_e/models/nn/encoding.py)
as its basis. Shap-E is distributed under the MIT License, copyright 2023
OpenAI. Its license text is included in `licenses/Shap-E-MIT.txt`.

## S-DON and DeepXDE

The `DeepONetCartesianProd` and `TripleCartesianProd` classes in the S-DON
research scripts adapt DeepXDE implementations, including their documentation.
Those adapted portions retain DeepXDE's LGPL-2.1 terms; they are not relicensed
by the root MIT file. The scripts depend on DeepXDE 1.9.3, whose license is
included verbatim in `licenses/DeepXDE-LGPL-2.1.txt`.

Upstream sources:

- [DeepXDE](https://github.com/lululxvi/deepxde)
- [TensorFlow DeepONet implementation at v1.9.3](https://github.com/lululxvi/deepxde/blob/v1.9.3/deepxde/nn/tensorflow/deeponet.py)
- [Cartesian-product dataset implementation at v1.9.3](https://github.com/lululxvi/deepxde/blob/v1.9.3/deepxde/data/triple.py)

The adapted network contracts the recurrent branch and spatial trunk with a
time-resolved tensor product; the dataset handles the corresponding field
history targets. Source provenance is recorded in `source_manifest.json`.
