[![Downloads](https://static.pepy.tech/badge/aestetik)](https://pepy.tech/project/aestetik)
# AESTETIK: AutoEncoder for Spatial Transcriptomics Expression with Topology and Image Knowledge

This model is part of the paper "Representation learning for multi-modal spatially resolved transcriptomics data".

**Authors**: Kalin Nonchev, Sonali Andani, Joanna Ficek-Pascual, Marta Nowak, Bettina Sobottka, Tumor Profiler Consortium, Viktor Hendrik Koelzer, and Gunnar Rätsch

The preprint is available [here](https://www.medrxiv.org/content/10.1101/2024.06.04.24308256v1).

### Do you want to gain a multi-modal understanding of key biological processes through spatial transcriptomics?

We introduce AESTETIK, a convolutional autoencoder model. It jointly integrates transcriptomics and morphology information, on a spot level, and topology, on a neighborhood level, to learn accurate spot representations that capture biological complexity.

![aestetik](/figures/aestetik.png)

**Fig. 1 AESTETIK integrates spatial, transcriptomics, and morphology information to learn accurate spot representations.**
**A**: Spatial transcriptomics enables in-depth molecular characterization of samples on a morphology and RNA level while preserving spatial location. **B**: Workflow of AESTETIK. Initially, the transcriptomics and morphology spot representations are preprocessed. Next, a dimensionality reduction technique (e.g., PCA) is applied. Subsequently, the processed spot representations are clustered separately to acquire labels required for the multi-triplet loss. Afterwards, the modality-specific representations are fused through concatenation and the grid per spot is built. This is used as an input for the autoencoder. Lastly, the spatial-, transcriptomics-, and morphology-informed spot representations are obtained and used for downstream tasks such as clustering, morphology analysis, etc.

## Setup

We can install aestetik directly through pip.

```
pip install aestetik
```

We can also create a conda environment with the required packages.

```
conda env create --file=environment.yaml
```

We can also install aestetik offline.

```
git clone https://github.com/ratschlab/aestetik
cd aestetik
python setup.py install
```

##### NB: Please ensure you have installed [pyvips](https://github.com/libvips/pyvips) depending on your machine's requirements. We suggest installing pyvips through conda:
```
conda install conda-forge::pyvips
```

## Getting Started

Please take a look at our [example](example/gettingStartedWithAESTETIK.ipynb) to get started with AESTETIK. [Here](example/gettingStartedWithAESTETIKwithSimulatedData.ipynb), another example notebook with [simulated spatial transcriptomics data](https://github.com/ratschlab/simulate_spatial_transcriptomics_tool).

![aestetik](/figures/AESTETIK_clustering.png)


## Citation

In case you found our work useful, please consider citing us:

```
@article{nonchev2024representation,
  title={Representation learning for multi-modal spatially resolved transcriptomics data},
  author={Nonchev, Kalin and Andani, Sonali and Ficek-Pascual, Joanna and Nowak, Marta and Sobottka, Bettina and Tumor Profiler Consortium and Koelzer, Viktor Hendrik and Raetsch, Gunnar},
  journal={medRxiv},
  pages={2024--06},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Press}
}
```

The code for reproducing the paper results can be found [here](https://github.com/ratschlab/st-rep).

## Contact

In case, you have questions, please get in touch with [Kalin Nonchev](https://bmi.inf.ethz.ch/people/person/kalin-nonchev).
