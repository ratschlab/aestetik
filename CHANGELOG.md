# Changelog
All notable changes to this project will be documented in this file.

## 0.3.0 (May 2026) — sklearn-style API

### 1. Breaking
- `AESTETIK` now inherits from `sklearn.base.BaseEstimator`,
  `TransformerMixin`, `ClusterMixin`. The public methods follow the
  standard scikit-learn surface:
  - `fit(X)` returns `self`, no longer mutates `X`.
  - `transform(X)` returns the latent embedding as `ndarray`.
  - `predict(X)` returns cluster labels as `ndarray`.
  - `fit_transform(X)` and `fit_predict(X)` are inherited.
- Constructor parameters renamed to snake_case:
  - `nCluster` → `n_cluster`
  - `random_seed` → `random_state`
- Training-time configuration (`validation_split`, `early_stopping_params`,
  `used_obsm_*`, `used_obs_batch`, `num_repeats`) moved from per-call
  kwargs to constructor parameters; `early_stopping_params` is split
  into `early_stopping_patience` and `early_stopping_min_delta`.

### 2. Added
- `embedding_`, `labels_`, `losses_`, `model_`, `trainer_`,
  `transcriptomics_weight_`, `morphology_weight_`,
  `obsm_transcriptomics_dim_`, `num_input_channels_` fitted attributes
  (sklearn trailing-underscore convention).
- `__version__` now exposed via `importlib.metadata`.
- `get_params()` / `set_params()` work out of the box.

### 3. Migration
Old:
```python
model = AESTETIK(nCluster=7, morphology_weight=1.5)
model.fit(adata, validation_split=0.2)
model.predict(adata, cluster=True)        # mutates adata
```
New:
```python
model = AESTETIK(n_cluster=7, morphology_weight=1.5, validation_split=0.2)
model.fit(adata)                          # returns self
adata.obsm["AESTETIK"]       = model.transform(adata)
adata.obs["AESTETIK_cluster"] = model.predict(adata)
```

## 0.2.0 (June 2025) — Lightning rewrite

### 1. Removed
#### 1.1 Removed methods
- `AESTETIK.summary(<parameters>)`
- `AESTETIK.prepare_input_for_model(<parameters>)`: integrated into `fit`
- `AESTETIK.train(<parameters>)`: replaced by `fit`
- `AESTETIK.compute_spot(<parameters>)`: replaced by `predict`
- `AESTETIK.vizualize(<parameters>)`: renamed to `visualize` and moved to [`utils/utils_visualization.py`](src/aestetik/utils/utils_visualization.py) (previously `utils_vizualization.py`)

#### 1.2 Removed instantiation parameters for AESTETIK 
The following parameters are no longer required during instantiation. Most are now passed directly to the methods that require them:
- `adata`
- `device`
- `used_obsm_transcriptomics`
- `used_obsm_morphology`
- `used_obsm_combined`
- `save_emb`
- `img_path`
- `spot_diameter_fullres`
---
### 2. Added
#### 2.1 New methods
- `AESTETIK.fit(<parameters>)`
- `AESTETIK.predict(<parameters>)`
- `AESTETIK.fit_predict(<parameters>)`
#### 2.2 New instantiation parameters for AESTETIK
- `num_workers: int = 7`
---
### 3. Changed
#### 3.1 Updated tutorial notebooks
- Tutorial notebooks [gettingStartedWithAESTETIK.ipynb](example/gettingStartedWithAESTETIK.ipynb) and [gettingStartedWithAESTETIKwithSimulatedData.ipynb](example/gettingStartedWithAESTETIKwithSimulatedData.ipynb) have been updated to use the new API
---
### New capabilities
- **Cross-sample training and prediction:** Train on one sample, predict on another
- **Multi-sample support:** Train and predict across multiple samples
