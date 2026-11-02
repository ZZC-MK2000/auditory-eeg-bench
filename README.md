# Auditory EEG Bench (AEB) — v0.1

Repository: https://github.com/ZZC-MK2000/auditory-eeg-bench

**A benchmark for naturalistic auditory EEG decoding, not an encoder paper.**

AEB fixes the data cache, split suite, model zoo, controls and reporting format so that
"good auditory EEG representation" can be compared without changing the protocol between
papers. The reference task is NMED-T 10-way song identification; the stress test is
leave-one-dataset-out (LODO) over six auditory datasets.

> **Status.** v0.1 contains the fixed split files, a leakage audit, frozen-probe reference
> results for LaBraM/BIOT/CBraMod and a compact spatial-spectral encoder, from-scratch
> supervised reference results, and a hand-crafted band-power baseline in Appendix A.
> The unified 3-seed fine-tune grid and the full LODO leaderboard are the next release;
> this repository does not claim they are complete.

## What is fixed

| Component | Definition |
|---|---|
| Training cache | six datasets, 19 channels, 200 Hz, non-overlapping 8-s windows |
| Scale | **69,108 windows, 195 dataset-subject entries, 153.6 h** |
| Target task | NMED-T, 20 subjects × 10 songs, 10-way window identification |
| Primary readout | frozen features + per-subject logistic probe (trial-wise 50/50) |
| Secondary readout | supervised fine-tuning, same per-subject trial-wise splits |
| Split suite | random rows / held-out blocks / held-out subjects / leave-one-dataset-out |
| Controls | zero EEG, shuffled EEG, answer-conditioned query, adjacent-window upper bound |
| Reporting | chance/majority, mean ± SD, subject-level paired tests, 95% CI |
| Main comparison | frozen EEG foundation-model encoders (BIOT, LaBraM, CBraMod) + the six-dataset encoder |
| Feature baseline | hand-crafted band-power, Appendix A only |

## Split suite

All split files use row indices into local caches; no raw EEG is redistributed.
`nmedt_folds.json` has 3 seeds for each of random rows, blocks and subjects.
`leave_one_dataset_out.json` has one split per held-out dataset.

| Split | Unit | Expected leakage | Train / val / test (seed 0) |
|---|---|---|---:|
| random rows | window | subject + adjacent windows (intentional upper bound) | 4,914 / 1,053 / 1,053 |
| held-out blocks | 5-window contiguous block | no block overlap; same subjects | 4,883 / 1,071 / 1,066 |
| held-out subjects | subject | no subject overlap | 4,914 / 1,053 / 1,053 |
| leave-one-dataset-out | dataset | no dataset/subject overlap | varies |

The audit is in `manifests/split_audit.json`. Random-row subject overlap is 20/20 by
design; block splits also keep subjects in training; subject splits have zero subject
overlap. LODO splits have zero dataset and zero subject overlap.

## Reference results

Main comparison: frozen EEG foundation-model encoders, same per-subject trial-wise
logistic probe (5 split repetitions; chance 0.10). Our six-dataset encoder is evaluated
against the released BIOT, LaBraM and CBraMod checkpoints. Band-power is a hand-crafted
feature baseline, not a learned encoder, and is reported only in Appendix A.

| Model | Family | Params | 10-way | Centroid | Note |
|---|---|---:|---:|---:|---|
| Encoder (6 datasets) | frozen encoder | 1.14M | **0.607** | 0.462 | target-unseen encoder |
| BIOT | pretrained FM | 3.19M | 0.381 | 0.266 | 19→18-channel interpolation |
| LaBraM-200 | pretrained FM | — | 0.358 | 0.291 | frozen 200-d |
| CBraMod | pretrained FM | 4.92M | 0.250 | 0.218 | 19-channel |

EEGPT is excluded: its released checkpoint uses a 62-channel token layout, and a
19-channel adaptation would require a new channel projection rather than a comparable
frozen-feature run.

### Supervised from-scratch reference

These are not foundation-model encoders and are not part of the main comparison; they are
fixed-budget reference points, not tuned upper bounds.

| Model | Params | Per-subject | Pooled | Note |
|---|---:|---:|---:|---|
| EEGNet | 9K | 0.118 | 0.134 | from scratch, per-subject |
| ShallowConvNet | 72K | 0.258 | 0.231 | from scratch, per-subject |
| Deep4Net | 305K | 0.107 | 0.133 | from scratch, per-subject |
| EEG-Conformer | 1.19M | 0.142 | 0.120 | from scratch, per-subject |

Supervised models were also trained pooled across subjects (0.120–0.231). They are
reference points, not tuned upper bounds: the per-subject training half has ~17
windows/class and EEGNet reaches 1.00 train / 0.14 test on a representative subject.
A unified 3-seed, tuned fine-tune grid is listed as a release blocker below.

## Controls

- **Split-ladder control:** the band-power feature control is reported in Appendix A.
- **Answer-conditioned query** (released MindMix implementation, KUL S1): real, zero and
  shuffled EEG all report 1.000 pair accuracy; label-free pairwise .503 and EEG-only .501.
- **Zero/shuffled EEG** must be reported for every new query-conditioned method.
- **Adjacent windows** are retained only as an intentional upper bound, never as a
  cross-subject result.

## Data and licenses

The dataset table and redistribution rules are in `DATASHEET.md` and `DATA_LICENSES.md`.
In short: OpenNeuro ds002725 and MUSIN-G are CC0; SparrKULee is CC BY-NC 4.0 and is not
redistributed (download script and split files only); CIRE/KUL/DTU licenses must be
confirmed with their providers before any raw-data release. NMED-T stimuli are
copyrighted; only derived features, aggregate results, and stimulus spectrograms are released;
no stimulus waveform is redistributed.

## Reproduce

```bash
# 1. put local caches at the paths in configs/datasets.yaml
python code/audit_splits.py
python code/run_probe.py --features frozen_features.npz --split splits/nmedt_folds.json
python code/run_finetune.py --cache /path/to/nmed_cache --model eegnet --seeds 0 1 2
python code/make_leaderboard.py --results results/reference.json --out leaderboard/index.html
```

The cache builder and dataset access notes are in `DATASHEET.md`; raw EEG, stimulus audio
and checkpoints are not redistributed. The demo contains generated clips and stimulus
spectrograms only.

## Release blockers before calling the benchmark final

1. Unified 3-seed grid for every frozen and fine-tune model; report both separately.
2. Tuned supervised budget (validation-based early stopping or a preregistered grid).
3. LODO results for the full model zoo.
4. EEGPT: either a documented 62→19 channel projection or a formal exclusion in the
   leaderboard (currently excluded).
5. Subject-level paired tests and 95% CIs for every main-table claim.

## Appendix A — Hand-crafted feature baseline

Not a learned encoder and not part of the main comparison. Same splits; reported for
completeness because it is a strong closed-set label feature.

| Feature | Dim | 10-way | Centroid | Note |
|---|---:|---:|---:|---|
| Band-power 82×22 | 1,804 | **0.840** | 0.670 | hand-crafted feature baseline |
| Band-power 19×5 | 95 | 0.776 | 0.485 | 19-channel version |

Split ladder (band-power feature control): random rows .896 / blocks .873 / subjects .106.

## Demo

A stand-alone genre-selection reconstruction demo is in [`demo/`](demo/). Each row is one
held-out EEG window: the heard stimulus is shown as a spectrogram only (the recording is
copyrighted and not redistributed), the reconstructed clip is shown as a spectrogram plus
playable audio, and the CLAP cosine of the generated clip to its own stimulus versus the
other 13 stimuli is reported. The spectrogram tiles share an identical time axis (0–8 s),
frequency axis (0–8 kHz), and colour scale. The demo is illustrative, not a benchmark
result: no MOS was collected, and CLAP own/others is a demo-level check. The same-prompt
control is near zero, so the CLAP margin is genre-level rather than song-level.

## Citation

If you use AEB, cite the dataset sources and this benchmark release. See `CITATION.cff`.
