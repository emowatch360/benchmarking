# Model Training

This folder contains two model families: a multimodal encoder (`multimodal_enc`) and an XGBoost baseline (`xgboost`). Both share the same configuration system, data loading utilities, and cross-validation logic.

---

## Configuration

### `configs/shared_config.py`

Set `data_folder` to the root path of your dataset before running anything:

```python
data_folder = '/path/to/your/data/'
seed = 42
```

`config_folder` is resolved automatically to the directory of this file.

### `configs/multimodal_enc.yaml`

Key sections:

**`model`**
- `model_type`: encoder architecture — `cnn`, `resnet`, or `mha`
- `conv_layers`: channel sizes per conv layer for the current-period encoder
- `sp_conv_layers`: channel sizes for the sleep-period encoder
- `feature_split`: how features are split across modality groups (e.g. `[4, 4, 20]`)

**`training`**
- `algorithm`: domain generalisation algorithm — `ERM`, `IRM`, `VREX`, `DRO`, `DANN`, `HHISS`, `Siamese`
- `use_cp_ssl_pretraining` / `use_sp_ssl_pretraining`: whether to load a pretrained encoder
- `freeze_encoders`: freeze encoder weights during finetuning
- `freeze_then_finetune`: first train with frozen encoders, then unfreeze for remaining epochs

**`pretraining`**
- `pretraining_type`: SSL objective — `contrastive-ntxent`, `contrastive-pairwise`, `contrastive-loo`, `regularization-vicreg`

**`data`**
- `cv`: cross-validation scheme — `loso`, `walk-forward`, `time-series`
- `feature_dim`: `2D` (per-minute time series) or `1D-stat` (statistical summaries)
- `feature_type_list`: subset of modalities to use, or `['all']`
- `include_sleep_features`, `include_sleep_stats`, `include_demo_info`, `include_time_index`: optional feature groups to concatenate

### `configs/xgboost.yaml`

Standard XGBoost hyperparameters under `training`. Data flags mirror those in `multimodal_enc.yaml` but only `1D-stat` features are supported.

---

## Running

### Multimodal encoder — finetuning

Edit and run `multimodal_enc/evaluate_and_save_results.py`. The script loops over labels, algorithms, architectures, and feature combinations and calls `perform_multimodal_enc_training(...)`:

```bash
cd model_training
python -m multimodal_enc.evaluate_and_save_results
```

### Multimodal encoder — SSL pretraining only

```bash
python -m multimodal_enc.main_pretrain
```

Pretrained encoder weights are saved to the numpy pretraining folder. Set `use_cp_ssl_pretraining: True` (and optionally `use_sp_ssl_pretraining: True`) in `multimodal_enc.yaml` before running finetuning to load them.

### XGBoost

```bash
cd model_training/xgboost
python evaluate_and_save_results.py
```
