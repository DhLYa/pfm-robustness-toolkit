# pfm-robustness-toolkit

Robustness benchmarking for pathology foundation models (PFMs) under clinically realistic
image perturbations. Built on the Robustness Evaluation and Enhancement Toolbox
([REET](https://github.com/alexjfoote/reetoolbox)).

This is the code behind my MSc Computer Science dissertation at the University of Warwick
(2025, supervised by Professor Fayyaz Minhas):

> **Beyond Accuracy: Robustness Benchmarking of Pathology Foundation Models under Clinically Realistic Perturbations**

The work benchmarks 13 Vision Transformer PFMs against 2 CNN baselines (ResNet18 and ResNet50)
on 4 patch-level histopathology datasets. It measures how classification performance
degrades under 11 perturbations, such as stain variation, blur, JPEG compression and
geometric changes, and under distribution shift built from the data itself.

## Status

The research code from the dissertation has been refactored into a package
(`pfm_robustness`) with a central registry of models and datasets and one short notebook per
pipeline stage. Adding a new model or dataset is now a single registry entry (see
[Adding a model or dataset](#adding-a-model-or-dataset)).

The refactor keeps the numerical behaviour of the dissertation runs, including some known
quirks (see [Known issues](#known-issues)). The original notebooks are kept for reference in
`notebooks/archive/pre_refactor/`.

## What this adds to REET

REET provides the perturbation transforms, the adversarial (PGD) and stochastic search
optimisers, and the input sensitivity and fooling ratio metrics. This repository adds:

- **Model loaders for 15 models** through Hugging Face and timm. Each PFM is used as a frozen
  feature extractor with a single linear classification head, so the comparison reflects
  the pre-trained representations rather than head design.
- **Dataset loaders** for NCT-CRC-HE, PANDA, PanNuke and PatchCamelyon, each recast as binary
  patch classification.
- **Fixed-parameter evaluation**: every model under every perturbation at a set,
  clinically plausible strength. Runs are repeated 3 times, results go to per-model Excel
  files, and misclassified images are saved for visual comparison.
- **Constraint sweeps**: perturbation strength is stepped from zero to a maximum, with
  ROC-AUC recorded at each step over 3 repeat runs.
- **Perturbation Performance Index (PPI)**: a single robustness score per model and
  perturbation, computed from the sweep curves (defined below).
- **Embedding analysis**: train-test cosine similarity of feature embeddings as
  perturbation strength increases, plus t-SNE views of how embeddings move.
- **NR-Kfold distribution shift**: folds built by clustering embeddings, so each held-out
  fold is dissimilar to the training folds, evaluated with a linear probe.
- **Reporting**: annotated heatmaps, sweep curves with standard deviation bands, PPI bar
  charts, PPI heatmaps, PPI against parameter count, and cross-dataset rankings.

## Perturbation Performance Index (PPI)

For a sweep of K constraint values:

```
PPI = (1/K) * sum over k of ROC-AUC(k)
```

This is the normalised area under the ROC-AUC against perturbation strength curve. In the
code it is computed per run as the trapezoidal area under ROC-AUC against the min-max
normalised constraint, then averaged over runs. A model that keeps perfect discrimination
across the whole sweep scores 1, and the score falls as performance degrades. PPI depends on
the sweep range chosen, so it is a comparative metric across models evaluated on the same
sweep, not an absolute one.

## Datasets

All four datasets are public and anonymised. They are not included in this repository.

| Dataset | Task (binary) | Notes |
|---|---|---|
| [NCT-CRC-HE-100K / CRC-VAL-HE-7K](https://zenodo.org/records/1214456) | Tumour epithelium vs all other tissue classes | 224×224 patches at 0.5 MPP, Macenko normalised, about 1:8 class ratio. The 7K set is the independent test set. Subsets were used because of compute limits. |
| [PANDA](https://github.com/MicheleDamian/prostate-gleason-dataset) (processed subset) | Benign (Gleason 0) vs cancer (Gleason 3, 4 or 5) | Only slides with matching primary and secondary patterns were kept. 512×512 patches with stride 256. Roughly balanced. |
| [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) | Neoplastic vs non-neoplastic patches | Positive means at least 5 neoplastic nuclei and negative means none. Patches with 1 to 4 are excluded. Uses the dataset's 3 predefined folds. |
| [PatchCamelyon](https://github.com/basveeling/pcam) | Lymph node metastasis present vs absent | About 330,000 patches from Camelyon16. The predefined splits have no slide overlap. |

The loaders expect this layout under `Data/`:

```
Data/
  NCT/NCT-CRC-HE-100K/<CLASS>/*.tif            training set (ADI, BACK, ..., TUM)
  NCT/CRC-VAL-HE-7K/<CLASS>/*.tif              test set
  PANDA/train.csv, PANDA/test.csv              columns: image, class
  PANDA/train/<image>, PANDA/test/<image>
  PanNuke/Fold {1,2,3}/images/fold{k}/images.npy
  PanNuke/Fold {1,2,3}/masks/fold{k}/masks.npy
  PatchCamelyon/images/camelyonpatch_level_2_split_{train,valid,test}_{x.h5,meta.csv}
```

The splits are:

- **NCT**: train on 100K, test on 7K.
- **PanNuke**: train on folds 1 and 2, test on fold 3.
- **PANDA**: the train and test CSVs.
- **PatchCamelyon**: the official train, valid and test splits.

Each pipeline stage uses a stratified subsample. The fractions used for the dissertation are
set per dataset and per stage in `DATASETS[...].budgets` in `pfm_robustness/datasets.py`.

## Models

| Model | Parameters | Source |
|---|---|---|
| ResNet18 | 11.2M | torchvision (ImageNet weights, fine-tuned per dataset) |
| ResNet50 | 23.5M | torchvision (ImageNet weights, fine-tuned per dataset) |
| Hibou-B | 85.7M | `histai/hibou-b` |
| H0-mini | 85.7M | `bioptimus/H0-mini` |
| EXAONEPath | 85.8M | `LGAI-EXAONE/EXAONEPath` |
| Phikon v1 | 86.4M | `owkin/phikon` |
| UNI | 303.4M | `MahmoodLab/uni` |
| Phikon v2 | 303.4M | `owkin/phikon-v2` |
| Hibou-L | 303.7M | `histai/hibou-L` |
| Virchow | 631.2M | `paige-ai/Virchow` |
| Virchow2 | 631.2M | `paige-ai/Virchow2` |
| UNI2 | 681.4M | `MahmoodLab/UNI2-h` |
| H-Optimus-0 | 1,134.8M | `bioptimus/H-optimus-0` |
| H-Optimus-1 | 1,134.8M | `bioptimus/H-optimus-1` |
| GigaPath | 1,135.0M | `prov-gigapath/prov-gigapath` |

Several of these are gated on Hugging Face. You need to request access on each model page and
either log in with `huggingface-cli login` or set `HF_TOKEN` before the weights will download.

## Perturbations

| Perturbation | Search | Simulates | Sweep range | Fixed value |
|---|---|---|---|---|
| Pixel | Adversarial (PGD) | Sensor or electronic noise | C: 0 to 0.05 | C = 0.03 |
| Stain | Adversarial | Stain variation between labs | not swept | C = 0.1 |
| Brightness | Adversarial | Illumination or scanner bias | C: 0 to 7.5 | C = 0.1 |
| HED | Stochastic | Stain variation in HED colour space | not swept | α 1 ± 0.075, β ± 0.075 |
| Random stain | Stochastic | Random stain re-mixing | 0 to 0.5 | ± 0.125 |
| JPEG | Stochastic | Lossy compression artefacts | quality 100 to 90 | quality 97.5 to 100 |
| Blur | Stochastic | Out-of-focus regions (Gaussian) | σ 1 to 7 | σ 3 to 20 on a random region |
| Rotate | Stochastic | Orientation differences | 0 to 360° | 0 to 360° |
| Crop | Stochastic | Occlusion or poor framing | 224 to 50 px | 50 to 150 px |
| Zoom in | Stochastic | Smaller field of view at the same input size | 1 to 3× | 1.1 to 2× |
| Zoom out | Stochastic | Reduced effective resolution | 1 to 0.05× | 0.5 to 0.95× |

Fixed-parameter strengths and sweep ranges were chosen by binary search, so that each
perturbation is visually noticeable and causes measurable degradation in every model. Sweeps
used about 20 steps (the exact counts are in `SWEEPS` in `pfm_robustness/perturbations.py`).

## Metrics

- **Baseline accuracy and ROC-AUC** on the unperturbed test set.
- **Input sensitivity** (REET): mean absolute change in model output after perturbation.
- **Fooling ratio** (REET): share of correctly classified samples that become misclassified.
- **Adversarial ROC-AUC and PR-AUC**: discrimination measured on perturbed samples.
- **PPI**: aggregate robustness across a constraint sweep.

## Results

### PPI averaged across all four datasets

| Avg. rank | Model | Avg. PPI | Avg. std |
|---|---|---|---|
| 2.75 | UNI2 | 0.771 | 0.003 |
| 4.00 | Phikon v2 | 0.761 | 0.003 |
| 4.25 | H-Optimus-1 | 0.758 | 0.003 |
| 6.50 | EXAONEPath | 0.736 | 0.003 |
| 6.50 | GigaPath | 0.738 | 0.003 |
| 6.75 | Hibou-L | 0.739 | 0.004 |
| 7.25 | H-Optimus-0 | 0.736 | 0.004 |
| 7.25 | Virchow2 | 0.726 | 0.003 |
| 7.75 | UNI | 0.733 | 0.003 |
| 8.25 | H0-mini | 0.723 | 0.004 |
| 9.50 | ResNet50 | 0.708 | 0.003 |
| 10.75 | Phikon v1 | 0.692 | 0.003 |
| 11.75 | Hibou-B | 0.681 | 0.004 |
| 12.25 | ResNet18 | 0.685 | 0.004 |
| 14.50 | Virchow | 0.638 | 0.004 |

### NR-Kfold ROC-AUC averaged across all four datasets

The NR-Kfold analysis clusters each model's embeddings with k-means within each class
(5 clusters per class). Whole clusters are then assigned to 5 folds, keeping each class
balanced across folds, so each held-out fold is deliberately dissimilar to the training
folds. A linear head is trained on 4 folds and tested on the fifth. Variation across folds
is the main robustness signal. H0-mini is excluded because embedding extraction failed for it.

| Avg. rank | Model | Avg. ROC-AUC | Avg. std |
|---|---|---|---|
| 4.25 | Virchow2 | 0.955 | 0.059 |
| 4.75 | H-Optimus-0 | 0.901 | 0.108 |
| 5.00 | Virchow | 0.910 | 0.090 |
| 5.50 | UNI2 | 0.914 | 0.068 |
| 6.25 | Hibou-L | 0.874 | 0.091 |
| 6.50 | Hibou-B | 0.842 | 0.146 |
| 7.00 | UNI | 0.901 | 0.095 |
| 7.75 | H-Optimus-1 | 0.889 | 0.121 |
| 8.00 | GigaPath | 0.873 | 0.102 |
| 8.75 | Phikon v1 | 0.858 | 0.126 |
| 9.75 | Phikon v2 | 0.811 | 0.164 |
| 10.00 | ResNet50 | 0.832 | 0.132 |
| 10.50 | EXAONEPath | 0.786 | 0.164 |
| 11.00 | ResNet18 | 0.787 | 0.121 |

### Findings

- **Baseline performance hides robustness differences.** Almost every model reaches ROC-AUC
  of about 0.99 or higher on NCT and PanNuke. The models only separate once they are
  perturbed.
- **Under perturbation, PFMs and ResNets are broadly comparable overall.** Rankings change
  with dataset and perturbation type, and no model is most robust everywhere.
- **ResNets are more robust to pixel noise.** This was the hardest perturbation for most
  models. PFMs are more robust to stain variation, rotation and zoom out.
- **Under data-derived distribution shift (NR-Kfold), PFMs generalise better.** The ResNets
  rank in the bottom third. PFM embeddings also form tighter, better-separated clusters in
  t-SNE.
- **Model size explains little.** The correlation between parameter count and mean PPI is
  0.13 (NCT), 0.17 (PANDA), 0.45 (PanNuke) and 0.43 (PatchCamelyon). For pixel noise it is
  around -0.5 on every dataset.
- **Virchow underperforms for its size**, ranking last on PPI on 3 of the 4 datasets.
- **Adversarial and stochastic perturbations behave differently in embedding space.**
  Adversarial pixel noise flips predictions while barely moving the embeddings. That
  suggests decision boundaries lie close to the data. Stochastic perturbations move
  embeddings steadily away from the training distribution.

## Repository layout

```
pfm_robustness/
    reet/                 REET core (transforms, PGD and stochastic search, Evaluator, metrics,
                          constraints, default parameters, differentiable JPEG)
    models.py             loaders for all 15 models, input normalisation, MODELS registry
    datasets.py           NCT, PANDA, PanNuke and PatchCamelyon datasets, DATASETS registry
    perturbations.py      fixed-parameter settings (FIXED_EVAL) and sweep ranges (SWEEPS)
    embedders.py          pretrained backbones for the embedding and NR-Kfold analyses
    paths.py              finds Data/, Models/ and Stats/ (no hard-coded paths)
    train.py              trains the classification head per model and dataset
    evaluate.py           fixed-parameter evaluation, writes Excel results
    sweep.py              constraint sweeps and aggregate sweep plots
    ppi.py                PPI tables and plots per dataset
    similarity.py         embedding similarity sweeps, t-SNE and similarity grids
    kfold.py              NR-Kfold fold construction, linear probe and summaries
    rankings.py           cross-dataset rankings
    plots.py              cross-dataset summary figures
notebooks/
    01_train.ipynb                    train classification heads per dataset
    02_fixed_eval.ipynb               fixed-parameter evaluation
    03_constraint_sweep.ipynb         constraint sweeps and PPI
    04_ppi_summary.ipynb              PPI rankings and heatmaps across datasets
    05_embedding_similarity.ipynb     embedding similarity sweeps and t-SNE
    06_nr_kfold.ipynb                 NR-Kfold fold construction, linear probe, rankings
    extras/                           PatchCamelyon-specific one-off analyses
    archive/                          original dissertation notebooks and WSI experiments
tests/test_registry.py                consistency checks for the registries
third_party/EXAONEPath/               EXAONEPath model code (not on PyPI)
Data/  Models/  Stats/                datasets, trained heads and results (not in git)
```

## Setup

```bash
pip install -e .
git clone https://github.com/LG-AI-EXAONE/EXAONEPath.git third_party/EXAONEPath
export HF_TOKEN=hf_...
```

This needs Python 3.10 or later and numpy 2. The full dependency list is in `pyproject.toml`.

A CUDA GPU is needed for practical runtimes. The full constraint sweep
took about 40 hours per dataset on the Warwick HPC cluster, plus about 18 hours for the
fixed-parameter metrics.

Paths are found automatically from the repository root, which holds `Data/`, `Models/` and
`Stats/`. If the data lives somewhere else, for example on cluster scratch storage, point to
it with:

```bash
export PFM_STORAGE=/path/to/storage    # folder containing Data/, Models/ and Stats/
```

## Running the pipeline

Run the notebooks in `notebooks/` in order. Each one has a short configure cell (dataset and
model names) followed by a single call into the package.

| Notebook | Needs | Writes |
|---|---|---|
| `01_train` | data | `Models/<dataset>/<model>_<dataset>.pth` |
| `02_fixed_eval` | 01 | `Stats/<dataset>/<model>/<model>_stats.xlsx` and misclassified examples |
| `03_constraint_sweep` | 01 | sweep results per model, then `Stats/<dataset>/constraint_sweep/ppi_aggregate.xlsx` |
| `04_ppi_summary` | 03 | `Stats/PPI_aggregate_model_rankings.xlsx`, PPI heatmaps |
| `05_embedding_similarity` | data | `Stats/<dataset>/<model>/embeddings/`, similarity grid |
| `06_nr_kfold` | data | `Stats/<dataset>/<model>/distribution_shift/`, k-fold summaries and rankings |

The same functions can be called from a script, for example in a SLURM job:

```python
from pfm_robustness.sweep import run_sweeps
run_sweeps("NCT", ["UNI2", "ResNet50"], runs=3)
```

Before a full run, try one model on one dataset with a small subset. Both `run_fixed_eval`
and `run_sweeps` accept `test_multiplier=` to override the dataset's default subset size.
The registries can be checked with `pytest tests/`.

## Adding a model or dataset

**A new model** needs two things in `pfm_robustness/models.py`:

1. A loader `load_<model>(n_classes, device)` that returns `(model, optimizer)`. For most
   backbones this means creating the backbone and returning
   `_attach_head_and_freeze(backbone, n_classes, device)`.
2. One entry in `MODELS`, giving the display name, the checkpoint file key, the loader, the
   input normalisation and the parameter count:

   ```python
   ModelSpec("MyPFM", "mypfm", load_mypfm, fixed_norm(IMAGENET_NORM), params_m=307.0),
   ```

   For normalisation, use `timm_norm` to read it from a timm model's config, or
   `processor_norm("org/repo")` for a Hugging Face image processor.

The model is then available in notebooks 01 to 04. To include it in the embedding and
NR-Kfold analyses (05 and 06), also add it to `KFOLD_EMBEDDERS`, `SIMILARITY_BACKBONES` and
`_PIL_TRANSFORMS` in `pfm_robustness/embedders.py`. `pytest tests/` flags anything missing.

**A new dataset** needs two things in `pfm_robustness/datasets.py`:

1. A PyTorch `Dataset` that returns `(image, label, name)` with labels 0 or 1, and exposes
   per-item labels as `.labels` (or `.samples[i]["label"]`). If it ships official train and
   validation splits, set `official_trainval_splits = ("train", "valid")` on the class.
2. One entry in `DATASETS`, giving the train and test folders under `Data/`, the constructor
   arguments for each split, and the subset size and batch size for each stage. Set
   `resize_to_224=True` if the patches are not 224×224.

**A new perturbation** is a `Perturbation` entry in `FIXED_EVAL`, plus a `Sweep` entry in
`SWEEPS` if it should be swept, both in `pfm_robustness/perturbations.py`.

## Known issues

These are kept so results match the dissertation; fixing them will change the numbers.

- **Input range.** The REET transforms assume pixel values of 0 to 255, but images are fed in
  as 0 to 1, so perturbation strengths are relative to that scale.
- **Virchow v1** drops 4 patch tokens when building its features, which may contribute to
  its low PPI.
- **NR-Kfold** uses Euclidean k-means on raw embeddings (not cosine distance), and the ResNet
  embeddings are ImageNet logits rather than penultimate-layer features.
- **Training** now uses the same preprocessing as evaluation for all models, so retrained
  heads may differ slightly from the dissertation's.

## Limitations

- Results come from 3 runs per configuration. More runs would allow formal significance
  tests between models.
- PPI precision depends on sweep step size, which was limited by compute.
- The evaluation is patch-level only. Whole-slide tasks are not covered yet.
- Some datasets were subsampled for compute reasons.

## Planned work

- Extend to whole-slide tasks such as metastasis detection, HER2 status and Gleason grading,
  using attention-based MIL (CLAM) and graph neural network aggregators. Early experiments
  are in `notebooks/archive/wsi/`.

## Acknowledgements and licence

Based on [REET](https://github.com/alexjfoote/reetoolbox) by Alex Foote et al., licensed
under the Apache License 2.0. This repository contains modifications and additions by Dhyey
Yajnik and is released under the same licence (see `LICENSE`). The differentiable JPEG code
in `pfm_robustness/reet/diffjpeg.py` is from [DiffJPEG](https://github.com/mlomnitz/DiffJPEG)
under the MIT licence. Datasets and model weights are subject to their own licences.

If you use REET, please cite:

> Foote, A., Asif, A., Rajpoot, N. and Minhas, F. (2022). REET: robustness evaluation and
> enhancement toolbox for computational pathology. *Bioinformatics*, 38(12), 3312–3314.
