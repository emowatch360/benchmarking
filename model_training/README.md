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
- `use_cp_ssl_pretraining` / `use_sp_ssl_pretraining`: use an SSL-pretrained encoder for the current-period / sleep-period data (pretrained on the fly, or loaded if a matching checkpoint already exists; see [SSL pretraining](#multimodal-encoder--ssl-pretraining))
- `use_pretrained_encoders`: always load existing pretrained encoder checkpoints instead of pretraining
- `freeze_encoders`: freeze encoder weights during finetuning
- `freeze_then_finetune`: first train with frozen encoders for `freeze_then_finetune_epochs` epochs, then unfreeze for the remaining epochs. Early stopping is not applied while encoders are frozen.
- `unfreeze_grace_epochs` (optional, default `0`): extra epochs after unfreezing before early stopping starts monitoring
- `continuous_val_tracking` (optional, default `False`): track the best validation loss from epoch 1, so the best frozen-phase model must be beaten by the unfrozen phase; the patience counter is reset when the frozen (and grace) phase ends

**`pretraining`**
- `pretraining_type`: SSL objective — `contrastive-ntxent`, `contrastive-pairwise`, `contrastive-loo`, `regularization-vicreg`

**`data`**
- `cv`: cross-validation scheme — `loso` or `walk-forward`
- `feature_dim`: `2D` (per-minute time series) or `1D-stat` (statistical summaries)
- `feature_type_list`: subset of modalities to use (`ppg`, `skin-temperature`, `acc`, `time-hrv`, `freq-hrv`, `non-linear-hrv`), or `['all']`. Works for both `2D` and `1D-stat` features; for `1D-stat`, selecting a feature selects all of its statistics (`mu`, `md`, `mx`, `mn`, `std`). Current-period and sleep-period features must share the same feature order.
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

`dataset_country` (`both`, `c1`, or `c2`) selects the dataset; results are written to `<results.root_dir>/<dataset_country>_multimodal_enc_...`.

With `save=True`, each CV fold writes the following to the results log folder:
- `eval_results_iter{k}.csv`: per-epoch train/validation losses and ROC
- `predictions_iter{k}.csv`: test-set predictions per sample (`pid`, `day_id`, `time_index`, class probabilities, predicted labels at the default and the validation-selected threshold, and the threshold itself)

### Multimodal encoder — SSL pretraining

SSL pretraining runs as part of finetuning when `use_cp_ssl_pretraining` and/or `use_sp_ssl_pretraining` are `True`. To avoid test-set leakage, encoders are pretrained separately for each CV fold:
- **LOSO**: pretraining uses only the fold's training and validation subjects; the held-out test subject is excluded.
- **Walk-forward**: pretraining uses only days strictly before the fold's test day.

The pretraining arrays (`<country>_emowatch_pretraining.npz` and `<country>_emowatch_pretraining_sleep.npz`) must therefore contain `pid_labels` and `day_labels` (produced by `dataset_creation`), with participant IDs consistent with the finetuning arrays.

Each fold starts finetuning from a fresh copy of its pretrained encoders. Checkpoints are saved to `multimodal_enc/pretraining/`:
- Pretrained: `encoder_<country>_<fold>_<cp|sp>_<pretraining_type>_<model_type><layers>_batch<B>_epoch<E>_feat_<features>_<i>_pretrained.pt`, where `<fold>` is `heldout<subject>` (LOSO) or `walkfwd_test<day>` (walk-forward)
- Finetuned (with `save=True`): `finetuned_encoder_<country>_<label>_<fold>_...`

If a matching pretrained checkpoint already exists, it is loaded instead of pretraining again. Delete the checkpoints to pretrain from scratch.

### XGBoost

```bash
cd model_training/xgboost
python evaluate_and_save_results.py
```
