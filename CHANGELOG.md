# Changelog
All notable changes to this project will be documented in this file.

## version (June 2025)
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
