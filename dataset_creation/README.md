# Dataset Creation (Finetuning) — Create NumPy/NPZ Files

This folder contains the code to convert the **saved per-minute feature CSVs** + **processed labels** into **NumPy arrays** for model finetuning.

The main entrypoint is:

- `create_all_data.py` → calls `load_emowatch_finetuning(...)` from `emowatch_utils.py`

Key files:
- `data_config.py` — defines the dataset dictionary (`ds_info`) and all paths / PID lists / feature lists
- `emowatch_utils.py` — implements the dataset assembly logic and saves `.npz`
- `create_all_data.py` — driver script that loops over epochs, feature dims, and labels

---

## What gets produced

Running `create_all_data.py` creates an `.npz` file per configuration (feature-dim × epoch length), saved in `ds_info[‘save-folder’]`
If `include_sleep=True`, the saved filename is modified to include sleep features

Each saved `.npz` contains a dictionary like:

- `features`: feature tensors (shape depends on `feature-dim`)
- `labels`: discrete class labels (binary by default in current config)
- `values`: raw continuous/likert values for the chosen label
- `session_labels`: survey session string per sample (`morning`/`afternoon`/`evening`/`bedtime`)
- `time_indices`: the **survey time index** (minute index in day) used as the endpoint of the window
- `prev_survey_values`: previous survey’s raw value (used for change-score style experiments)
- `day_labels`: day index (integer) corresponding to `wake_day`
- `pid_labels`: numeric PID index (1..N)
- `str_pid_labels`: string PID (e.g., `MM_01`, `pid03`)
- `dg_labels`: demographic feature vector (per sample)
- `feature_names`: expanded feature names (depends on `feature-dim`)

If `include_sleep=True`, it also includes:
- `sleep_statistics`: per-night sleep summary stats (e.g., duration, stage percentages)
- `sleep_features`: sleep-only features for that `wake_day`
- `sleep_statistic_names`
- `sleep_feature_names`

---

## Inputs required

Dataset creation assumes that preprocessing has already generated:

1) **Per-minute features** (one CSV per pid/day)
2) **Processed label table**
3) **Demographics table**
4) (Optional, if `include_sleep=True`) **Sleep features + sleep summary**
For C2 only, date continuity checks use `ds_info['c2-dates-path]`
All of the above paths are defined in `data_config.py`