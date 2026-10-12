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

The code here is the research code used for the dissertation. A refactor is in progress to
make configuration easier for new datasets, models and perturbations. The non-redundant
k-fold (NR-Kfold) fold construction code is not in the repository yet. It will be added with
the refactor.

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
- **Reporting**: annotated heatmaps, sweep curves with standard deviation bands, PPI bar
  charts, PPI heatmaps and PPI against parameter count.

## Perturbation Performance Index (PPI)

For a sweep of K constraint values:

```
PPI = (1/K) * sum over k of ROC-AUC(k)
```

This is the normalised area under the ROC-AUC against perturbation strength curve. A model
that keeps perfect discrimination across the whole sweep scores 1, and the score falls as
performance degrades. PPI depends on the sweep range chosen, so it is a comparative metric
across models evaluated on the same sweep, not an absolute one.

## Datasets

All four datasets are public and anonymised. They are not included in this repository.

| Dataset | Task (binary) | Notes |
|---|---|---|
| [NCT-CRC-HE-100K / CRC-VAL-HE-7K](https://zenodo.org/records/1214456) | Tumour epithelium vs all other tissue classes | 224×224 patches at 0.5 MPP, Macenko normalised, about 1:8 class ratio. The 7K set is the independent test set. Subsets were used because of compute limits. |
| [PANDA](https://github.com/MicheleDamian/prostate-gleason-dataset) (processed subset) | Benign (Gleason 0) vs cancer (Gleason 3, 4 or 5) | Only slides with matching primary and secondary patterns were kept. 512×512 patches with stride 256. Roughly balanced. |
| [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) | Neoplastic vs non-neoplastic patches | Positive means at least 5 neoplastic nuclei and negative means none. Patches with 1 to 4 are excluded. Uses the dataset's 3 predefined folds. |
| [PatchCamelyon](https://github.com/basveeling/pcam) | Lymph node metastasis present vs absent | About 330,000 patches from Camelyon16. The predefined splits have no slide overlap. |

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
log in with `huggingface-cli login` before the weights will download.

## Perturbations

| Perturbation | Search | Simulates |
|---|---|---|
| Pixel | Adversarial (PGD) | Sensor or electronic noise |
| Stain | Adversarial | Stain variation between labs |
| Brightness | Adversarial | Illumination or scanner bias |
| HED | Stochastic | Stain variation in HED colour space |
| Random stain | Stochastic | Random stain re-mixing |
| JPEG | Stochastic | Lossy compression artefacts |
| Blur | Stochastic | Out-of-focus regions (Gaussian) |
| Rotate | Stochastic | Orientation differences |
| Crop | Stochastic | Occlusion or poor framing |
| Zoom in | Stochastic | Smaller field of view at the same input size |
| Zoom out | Stochastic | Reduced effective resolution |

Fixed-parameter strengths and sweep ranges were chosen by binary search, so that each
perturbation is visually noticeable and causes measurable degradation in every model. Sweeps
used about 20 steps.

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

The NR-Kfold analysis clusters each model's embeddings with per-class k-means (cosine
distance, 5 clusters per class). Matching clusters across classes are combined into folds,
so each held-out fold is deliberately dissimilar to the training folds. A linear head is
trained on 4 folds and tested on the fifth. Variation across folds is the main robustness
signal. H0-mini is excluded because embedding extraction failed for it.

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
reetoolbox/
    transforms.py           perturbation transforms (from REET, extended)
    optimisers.py           PGD and stochastic search (from REET)
    image_evaluator.py      Evaluator that applies an optimised perturbation to a model
    metrics.py              accuracy, input sensitivity, fooling ratio, (adversarial) ROC-AUC and PR-AUC
    constants.py            default optimiser and transform parameters per perturbation
    constraints.py          constraint handling for optimisers
    loadmodels.py           loaders for all 15 models (frozen PFM backbones with a linear head)
    dataloaders.py          NCT, PANDA, PanNuke and PatchCamelyon datasets
    trainer.py              training loop
    eval_funcs.py           fixed-parameter evaluations, constraint sweeps, PPI and plotting
    run_all_model_stats.py  runs every model against every perturbation, writes Excel results
    utils.py                differentiable JPEG and general helpers
notebooks/
    train_models.ipynb                   train classification heads per dataset
    evals.ipynb                          fixed-parameter evaluation
    sweep_perturbs.ipynb                 constraint sweeps and PPI
    sweep perturbs for embeddings.ipynb  embedding similarity sweeps and t-SNE
```

## Setup

There is no packaged install yet. The main dependencies are:

```
torch torchvision timm transformers huggingface_hub
numpy pandas scikit-learn matplotlib seaborn openpyxl
```

A CUDA GPU is needed for practical runtimes. In the dissertation, the full constraint sweep
took about 40 hours per dataset on the Warwick HPC cluster, plus about 18 hours for the
fixed-parameter metrics.

Each notebook starts with a `PATH` variable pointing at the project root, which holds the
`data/` directory. Set it before running.

## Limitations

- Results come from 3 runs per configuration. More runs would allow formal significance
  tests between models.
- PPI precision depends on sweep step size, which was limited by compute.
- The evaluation is patch-level only. Whole-slide tasks are not covered yet.
- Some datasets were subsampled for compute reasons.

## Planned work

- Refactor for easier configuration of new models, datasets and perturbations.
- Add the NR-Kfold fold construction code.
- Extend to whole-slide tasks such as metastasis detection, HER2 status and Gleason grading,
  using attention-based MIL (CLAM) and graph neural network aggregators.

## Acknowledgements and licence

Based on [REET](https://github.com/alexjfoote/reetoolbox) by Alex Foote et al., licensed
under the Apache License 2.0. This repository contains modifications and additions by Dhyey
Yajnik and is released under the same licence (see `LICENSE`).

If you use REET, please cite:

> Foote, A., Asif, A., Rajpoot, N. and Minhas, F. (2022). REET: robustness evaluation and
> enhancement toolbox for computational pathology. *Bioinformatics*, 38(12), 3312–3314.
