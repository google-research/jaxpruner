# JaxPruner: a research library for sparsity research

<img src="https://github.com/google-research/jaxpruner/blob/main/images/jaxpruner.gif" alt="Jaxpruner logo " width="40%" align="middle">

Paper: [arxiv.org/abs/2304.14082](https://arxiv.org/abs/2304.14082)

## Introduction
*JaxPruner*, an open-source JAX-based pruning and sparse training library for machine learning research. *JaxPruner* aims to accelerate research on sparse neural networks by providing concise implementations of popular  pruning  and  sparse  training  algorithms  with  minimal  memory  and  latency overhead. Algorithms implemented in *JaxPruner* use a common API and work seamlessly with the popular optimization library Optax, which, in turn, enables easy integration with existing JAX based libraries. We demonstrate this ease of integration by providing examples in three different codebases: [scenic](https://github.com/google-research/scenic), [t5x](https://github.com/google-research/t5x) and [dopamine](https://github.com/google/dopamine) and [fedjax](https://github.com/google/fedjax).

We believe a sparsity library in Jax has the potential to accelerate sparsity research. This is because:

- Functional nature of jax makes it easy to modify parameters and masks.
- Jax is easy debug.
- Jax libraries and their usage in research is increasing.
- For further motivation read [why Deepmind uses jax here](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research).

There are exciting developments for accelerating sparsity in neural networks (K:N sparsity, CPU-acceleration, activation sparsity) and various libraries aim to enable such acceleration (todo). *JaxPruner* focuses mainly on accelerating algorithms research for sparsity. We mock sparsity by using binary masks and use dense operations for simulating sparsity. In the longer run, we also plan to provide integration with the [jax.experimental.sparse](https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html), aim to reduce the memory footprint of our models.

*JaxPruner* has 3 tenets: 
- **Easy Integration**: requires minimal changes to use.
- **Research First**: provides strong baselines and is easy to modify.
- **Minimal Overhead**: runs as fast as (dense) baseline.

### Easy Integration
Research in Machine Learning is fast paced.  This and the huge variety of Machine Learning applications result in a high number of ever-changing codebases. At the same time,adaptability of new research ideas highly correlates with their ease of use.  Therefore, *JaxPruner* is designed to be easily integrated into existing codebases with minimal changes.*JaxPruner* uses the popular [optax](https://github.com/deepmind/optax) optimization library to achieve this, requiring minimal changes when integrating with existing libraries.  State variables (i.e.  masks, counters) needed for pruning and sparse training algorithms are stored together with the optimization state, which makes parallelization and checkpointing easy.

```python
tx, params = _existing_code()
pruner = jaxpruner.MagnitudePruning(**config) # Line 1: Create pruner.
tx = pruner.wrap_optax(tx) # Line 2: Wrap optimizer.
```

### Research First
Often research projects require running multiple algorithms and baselines and so they benefit greatly from rapid prototyping. *JaxPruner* achieves this by committing to a generic API shared among different algorithms, which in return makes it easy to switch between different algorithms. We provide implementations for common baselines and make them easy to modify. A quick overview of such features are discussed in our colabs.

### Minimal Overhead
Sparse training and various pruning recipes requires some additional operations like masking. When we implement such basic operations we aim to minimize the overhead introduced (both memory and compute) and be as fast as the dense baseline.

```python
pruner = jaxpruner.MagnitudePruning(is_packed=True) # Line 1: Reduces mask overhead.
```
## Installation
You can install *JaxPruner* using pip directly from the source. 

```bash
pip3 install 
```

Alternatively you can also clone the source and run tests using the run.sh script.

```bash
git clone https://github.com/google-research/jaxpruner.git
cd jaxpruner
```

Following script creates a virtual environment and installs the necessary libraries. Finally, it runs the tests.

```bash
bash run.sh
```

## Quickstart
See our Quickstart colab:
[![Quick Start Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/jaxpruner/blob/main/colabs/quick_start.ipynb)

We also have [Deep-Dive](https://colab.research.google.com/github/google-research/jaxpruner/blob/main/colabs/deep_dive.ipynb) and [Mnist Pruning](https://colab.research.google.com/github/google-research/jaxpruner/blob/main/colabs/mnist_pruning.ipynb) colabs.

## Baselines
Here we share our initial experiments with baselines implemented.

|        |   no_prune |     random |   magnitude |   saliency |   global_magnitude |   magnitude_ste |   static_sparse |        set |       rigl |
|:-------|-----------:|-----------:|------------:|-----------:|-------------------:|----------------:|----------------:|-----------:|-----------:|
| ResNet-50 |   76.67    |   70.192   |    75.532   |   74.93    |           75.486   |         73.542  |        71.344   |   74.566   |   74.752   |
| ViT-B/16 (90ep)  |   74.044   |   69.756   |    72.892   |   72.802   |           73.598   |         74.208  |        64.61    |   70.982   |   71.582   |
| ViT-B/16 (300ep) |   74.842   |   73.428   |    75.734   |   75.95    |           75.652   |         76.128  |        70.168   |   75.616   |   75.64    |
| Fed. MNIST    |  86.21 |  83.53      |     85.74             |       85.60          |     86.01             |     86.16       |   83.33   |     84.20 |  84.64 |
| t5-Base (C4)   |    2.58399 |    3.28813 |     2.95402 |    3.52233 |            5.43968 |          2.7124 |         3.17343 |    3.13115 |    3.12403 |
| DQN-CNN (MsPacman)    | 2588.82    | 1435.29    |  2123.83    |  -       |         2322.21    |        -      |      1156.69    | 1723.3     | 1535.19    |
## Citation

```
@inproceedings{jaxpruner,
  title={JaxPruner: A concise library for sparsity research},
  author={Joo Hyung Lee and Wonpyo Park and Nicole Mitchell and Jonathan Pilault and Johan S. Obando-Ceron and Han-Byul Kim and Namhoon Lee and Elias Frantar and Yun Long and Amir Yazdanbakhsh and Shivani Agrawal and Suvinay Subramanian and Xin Wang and Sheng-Chun Kao and Xingyao Zhang and Trevor Gale and Aart J. C. Bik and Woohyun Han and Milen Ferev and Zhonglin Han and Hong-Seok Kim and Yann Dauphin and Karolina Dziugaite and Pablo Samuel Castro and Utku Evci},
  year={2023}
}
```

## Disclaimer
This is not an officially supported Google product.
