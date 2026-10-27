# MP2 submission: vote-aware listwise ranking over answer clusters

DASE7506 Mini-Project 2 · Author: Gao Zhe, student ID 3036707422

**Test accuracy 83.51% (390/467)**, from one run of the frozen method with the official evaluator. Prediction plus scoring takes 0.81 s on CPU with four threads, and the inference assets total 10,316 bytes.

The method has two parts:

1. Two L2 logistic probes read the supplied 1024-d embeddings. One sees the raw embedding; the other sees it minus the mean of its group ("centred").
2. A linear listwise ranker combines the probe scores with 30 handcrafted text features. It scores each candidate, pools the scores of candidates that share a parsed answer with log-sum-exp, and returns a member of the best-scoring answer cluster.

The report explains the design, the ablations and the failure analysis: [`report/MP2 Report.pdf`](report/MP2%20Report.pdf) (9 pages).

The course package (task guide, data, official runner and scorer) is unchanged. See `guide/GUIDE.md` and `code/README.md`.

## What was added to the course package

| Path | Purpose |
|---|---|
| `code/submission.py` | `predict(bank, arrays, checkpoint)`: loads the bundle, extracts features, scores candidates and picks a cluster |
| `code/features.py` | 30 handcrafted features from candidate text only, never from references |
| `code/model.py` | NumPy probes, listwise ranker, cluster selection, bundle loading |
| `code/textmodel.py` | hashed n-gram text probe (ablation only; not part of the frozen model) |
| `code/train.py` | trains on the training split only and writes `runs/model_bundle/` |
| `code/experiments.py` | baselines, ablations and the five-seed check on train CV and validation |
| `code/diagnostics.py` | gap analysis, cross-feature statistics, length-matched pairs, error structure, failure examples |
| `code/make_figures.py` | report figures from the stored records |
| `code/runs/model_bundle/` | **the checkpoint**: `config.json` and `weights.npz` |
| `code/runs/*.json`, `code/runs/experiments/`, `code/runs/diagnostics/` | evaluation JSON, freeze record, experiment and diagnostic records |
| `report/` | the report (`MP2 Report.pdf`, source `MP2 Report.docx`) and the figures used in it (`report/figures/paper/`) |

## Installation

Python 3.12. Run everything from `code/`.

```bash
cd code
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_data.py
```

`requirements.txt` pins `numpy==2.5.3`, the only dependency for training and inference. The figures also need `pip install -r requirements-report.txt`, which adds `matplotlib==3.11.2`.

**Windows.** `check_data.py` compares the data files byte for byte with their recorded hashes, so a line-ending conversion breaks it. This repository's `.gitattributes` marks `code/data/**` as `-text`, so Git never converts those files and a normal clone passes, whatever `core.autocrlf` is set to. The original course repository has no such file. A Windows clone of it with `core.autocrlf=true` fails the check unless conversion is turned off:

```bash
git clone -c core.autocrlf=false <course repository URL>
```

## Reproduce the test score without retraining

Use the checkpoint as committed. Run with four threads, as the guide requires.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python predict.py --method submission.py --checkpoint runs/model_bundle --split test --output runs/test-predictions.json
```

```bash
python score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

On Windows PowerShell, set the three variables first, for example `$env:OMP_NUM_THREADS='4'`. The expected result is `"correct": 390, "n": 467, "accuracy": 83.5117773019272`. The official output is committed as `code/runs/test-score.json`.

Replace `--split test` with `--split validation` for the validation score: 746/798, 93.48%.

## Retrain the checkpoint

```bash
python train.py
```

Training is deterministic: seed 7506, full-batch, CPU only, about 3 s. It writes `runs/model_bundle/` and reproduces `weights.npz` bit for bit. The frozen configuration is the default in `train.py` and is also recorded in `code/runs/freeze_record.json`.

| Asset | Bytes | SHA-256 |
|---|---|---|
| `runs/model_bundle/config.json` | 458 | `d2fd1db4d6f244750951a5df2ab958663b2d04da73b387349792933a77ecc656` |
| `runs/model_bundle/weights.npz` | 9,858 | `c0e4d448dcf945e0e25a2b892b09f309bf9d69a889f12e3fb5b618f58ef267ce` |

Submitted code version: commit `b3a1d38a708cf6e0570ad755bc5f12ca83297a27` on branch `mp2-submission`; the checkpoint is `code/runs/model_bundle/` in that commit.

## Reproduce the report's numbers

These commands use the training and validation splits only.

```bash
python experiments.py --seeds
```

```bash
python diagnostics.py
```

```bash
python make_figures.py --paper --output ../report/figures/paper
```

`experiments.py --seeds` takes about 12 minutes and writes `runs/experiments/results.json`. `diagnostics.py` writes `runs/diagnostics/`. `make_figures.py --paper` writes the figures used in the report, which have no in-figure titles and a white background. Without `--paper` it writes titled versions to `report/figures/`, which are not committed.

## Resources

| Item | Value |
|---|---|
| Inference assets | 10,316 bytes (limit 16 MiB) |
| Test prediction plus scoring | 0.683 s + 0.124 s = 0.807 s, measured externally (limit 30 s) |
| Local hardware | AMD Ryzen 7 9800X3D, Windows 11, OMP/OPENBLAS/MKL threads = 4. The reference machine is the Xeon Platinum 8457C |
| Training | about 3 s on CPU, seed 7506. No GPU used |
| Full experiment suite | about 12 min on CPU |

## Protocol

- **Training.** Models were fitted on training inputs and references only.
- **Development.** Development and method selection used the validation split and five-fold cross-validation on training.
- **Test.** The test bank was scored once, after the method was frozen. Hashes of the code and assets were recorded before and after that run (`code/runs/freeze_record.json`, `code/runs/freeze_hashes_pre_test.json`).
- **Not used.** No external data, no pretrained weights, no generation or API calls, and no question IDs.
- **New banks.** The method accepts a bank of any size.

## Reuse and data attribution

- **Data.** Questions and references derive from [GSM8K](https://github.com/openai/grade-school-math) (MIT license, retained in `code/data/LICENSE.txt` and `code/data/NOTICE.txt`). Candidate solutions and fixed embeddings were produced by the course staff with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), revision `c1899de289a04d12100db370d81485cdf75e47ca`. No model weights are distributed or used.
- **Course code.** `common.py`, `predict.py`, `score_predictions.py` and `check_data.py` are the course's own files, used unchanged; their hashes match `code/PACKAGE_MANIFEST.json`.
- **Third-party code.** None beyond NumPy, plus Matplotlib for figures.

## AI use

AI was utilized to assist with coding, data processing, chart creation, and document organization. Under the author’s guidance, it handled the debugging of the code in this repository, compiled data summaries and charts, and drafted an outline for the report based on the recorded experimental results.
