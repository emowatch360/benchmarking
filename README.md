# EmoWatch Benchmarking Codebase

Models repository for the EmoWatch project. Contains preprocessing, dataset creation, and model training code for the paper **EmoWatch: Investigating the Impact of Sleep Physiology on In-the-Wild Affect Prediction**.

---

## Repository Structure

```
emowatch-models/
├── dataset_preprocessing/    # Load, clean, and save per-minute wearable + HRV features
├── dataset_creation/         # Assemble preprocessed CSVs into NumPy arrays for training
├── model_training/           # XGBoost baseline and multimodal encoder with DG and SSL
└── requirements.txt
```

See the `README.md` in each subdirectory for detailed documentation:
- `dataset_preprocessing/README.md`
- `dataset_creation/README.md`
- `model_training/README.md`

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/emowatch360/benchmarking.git
cd emowatch-models
```

### 2. Create and activate a conda environment

```bash
conda create -n emowatch python=3.12
conda activate emowatch
```

### 3. Install PyTorch with CUDA 12.4 support

```bash
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

`torch`, `torchaudio`, and `torchvision` are excluded from `requirements.txt` and must be installed via the PyTorch index as above. Adjust the CUDA version if your driver requires a different one.

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`torch.cuda.is_available()` should return `True` if a GPU is available and CUDA drivers are correctly installed.

---

## Configuration

### Data path

Open `model_training/configs/shared_config.py` and set `data_folder` to the root of your dataset:

```python
data_folder = '/path/to/your/data/'
```

This path is used by both the multimodal encoder and XGBoost training scripts.

### Preprocessing paths

Open `dataset_preprocessing/config.py` and set `ROOT_FOLDER` to the same root data directory. `dataset_country` must be set to `'jp'` or `'sg'` depending on which cohort you are processing. See `dataset_preprocessing/README.md` for full details.

### Dataset creation paths

`dataset_creation/data_config.py` derives all paths from `ds_info['root-folder']`. Update the `get_dataset_dict` function if your directory layout differs from the defaults.

---

## Pipeline Overview

The pipeline runs in three sequential stages.

### Stage 1 — Preprocessing

Reads raw wearable sensor CSVs and HRV indices, cleans and aligns them to a per-minute time axis, and saves one feature CSV per `(participant, day)`.

```bash
cd dataset_preprocessing
python save_features.py
```

For SG only, follow up with:

```bash
python update_step_log_using_intraday.py
```

To extract sleep-window features:

```bash
python save_sleep_features.py
```

### Stage 2 — Dataset creation

Assembles the per-minute feature CSVs and processed label tables into `.npz` arrays ready for model training.

```bash
cd dataset_creation
python create_all_data.py
```

For pretraining arrays:

```bash
python create_pretrained_data.py
```

To merge arrays from both cohorts into a combined dataset:

```bash
python create_all_merged_data.py
```

### Stage 3 — Model training

Configure `model_training/configs/multimodal_enc.yaml` (or `xgboost.yaml`), then run the driver script for the desired model.

**Multimodal encoder:**
```bash
cd model_training
python -m multimodal_enc.evaluate_and_save_results
```

**XGBoost:**
```bash
cd model_training/xgboost
python evaluate_and_save_results.py
```

See `model_training/README.md` for configuration options, available algorithms, and SSL pretraining instructions.

---

## Third-Party Code

The following external implementations were adapted and incorporated directly rather than used as submodules.

**HHISS** (`model_training/multimodal_enc/algorithms/hhiss.py`)
Adapted from [yxiao54/HHISS_IMWUT](https://github.com/yxiao54/HHISS_IMWUT). The core teacher-student logic is unchanged. DRO replaces IRM as the base algorithm.
> Xiao, Y., Sharma, H., Kaur, S., Bergen-Cico, D., & Salekin, A. (2025). Human Heterogeneity Invariant Stress Sensing. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 9(3), 1-42.

**SleepFM contrastive losses** (`model_training/multimodal_enc/pretraining/pretrain.py`)
Pairwise and leave-one-out contrastive loss implementations adapted from [zou-group/sleepfm-clinical](https://github.com/zou-group/sleepfm-clinical).
> Thapa, R., He, B., Kjaer, M. R., Moore IV, H., Ganjoo, G., Mignot, E., & Zou, J. Y. (2024, March). Sleepfm: Multi-modal representation learning for sleep across ecg, eeg and respiratory signals. In AAAI 2024 Spring Symposium on Clinical Foundation Models.

**CroSSL aggregator and VICReg loss** (`model_training/multimodal_enc/pretraining/crossl_utils.py`)
Adapted from [Nokia-Bell-Labs/CroSSL](https://github.com/Nokia-Bell-Labs/CroSSL).
> Deldari, S., Spathis, D., Malekzadeh, M., Kawsar, F., Salim, F. D., & Mathur, A. (2024, March). Crossl: Cross-modal self-supervised learning for time-series through latent masking. In Proceedings of the 17th ACM international conference on web search and data mining (pp. 152-160).