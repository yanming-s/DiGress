# Optimized Graph Transformers for Generation with DiGress

This repository is forked from the original repository of the paper [*DiGress: Discrete Denoising Diffusion for Graph Generation*](https://openreview.net/forum?id=UaAD-Nu86WX). In this repository, we extended the [ZINC](https://zinc15.docking.org/) dataset and Graph Transformer [models](https://github.com/Klasnov/gtv2) to test the performance of the models on generation tasks.


## Environment Setup

To set up the environment, follow the steps given in the [original repository](https://github.com/cvignac/DiGress).

When running the self-defined Graph Transformer models, the model setting configuration files (like `configs/model/discrete.yaml`) should be adjusted, in order to keep the magnitude of the model parameters in a reasonable range.


## ZINC Dataset Extension

Compared to the original repository, we added the ZINC dataset extension. The `Dataset` and `DatasetInfos` classes are implemented in the `src/datasets/zinc_dataset.py` file. We also added the YAML configuration file `configs/dataset/zinc.yaml` for the ZINC dataset.


## Results

### Generation on QM9 Dataset

Generation results with the DiGress model on a subset of the QM9 dataset (2000 training samples, 200 testing samples) with batch size 512. Models are trained for 500 epochs, generating 1000 samples to evaluate validity, uniqueness, and novelty. The GTv2-Weighted model uses hyperparameter $\alpha = 0.5$.

|     Network      | Test Loss | Valid | Unique | Novelty |
| :--------------: | :---------------: | :--------: | :--------------: | :--------------: |
| *DiGress* | *142.9952* | *89.60%* | *99.33%* | *99.78%* |
| GTv1 | **142.6156** | 78.10% | 99.87% | 99.87% |
| GTv2-Weighted | 146.3584 | 73.10% | 99.61% | 99.61% |
|  GTv2-Gated  | 143.9262 | 72.20% | 99.86% | 99.86% |
|  GTv2-Mixed  | 144.2760 | 82.80% | 99.88% | **99.88%** |
|   GTv2-FiLM   | 143.8812 | **83.40%** | **100.00%** | 99.52% |


### Generation on ZINC Dataset

Generation results with the DiGress model on a subset of the ZINC dataset (5000 training samples, 200 testing samples) with batch size 512. Models are trained for 1000 epochs, generating 1000 samples to evaluate validity and uniqueness. The GTv2-Weighted model uses hyperparameter $\alpha = 0.5$.

|     Network      | Test Loss | Valid | Unique |
| :--------------: | :---------------: | :--------: | :--------------: |
| *DiGress* | *233.9124* | *60.90%* | *100.00%* |
| GTv1 | 249.4620 | 53.80% | 100.00% |
| GTv2-Weighted | 253.3110 | 53.10% | 100.00% |
|  GTv2-Gated  | 271.0986 | 58.40% | 100.00% |
|  GTv2-Mixed  | **240.4660** | 52.50% | 100.00% |
|   GTv2-FiLM   | 260.0413 | **60.60%** | 100.00% |


## References

1. [Vijay Prakash Dwivedi and Xavier Bresson. "A generalization of transformer networks to graphs." *arXiv preprint arXiv:2012.09699* (2020).](https://arxiv.org/abs/2012.09699)
2. [Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. "Digress: Discrete denoising diffusion for graph generation." *arXiv preprint arXiv:2209.14734* (2022).](https://arxiv.org/abs/2209.14734)
3. [Jonathan Ho, Ajay Jain and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
4. [Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin and Aaron Courville. "Film: Visual reasoning with a general conditioning layer." In Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1. 2018.](https://ojs.aaai.org/index.php/AAAI/article/view/11671)
