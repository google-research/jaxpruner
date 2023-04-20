# JaxPruner: a research library for sparsity research

## Introduction
JaxPruner, an open-source JAX-based pruning and sparse training library for machine learning research.JaxPruner aims to accelerate research on sparse neural networks by providing concise implementations of popular  pruning  and  sparse  training  algorithms  with  minimal  memory  and  latency overhead.  Algorithms implemented in JaxPruner use a common API and work seamlessly with the popular optimization library Optax, which, in turn, enables easy integration with existing JAX based libraries.  We demonstrate this ease of integration by providing examples in three different codebases:  [scenic](https://github.com/google-research/scenic), [t5x](https://github.com/google-research/t5x) and [dopamine](https://github.com/google/dopamine).

We believe a sparsity library in Jax has the potential to accelerate sparsity research. This is because:

- Functional nature of jax makes it easy to modify parameters and masks.
- Jax is easy debug.
- Jax libraries and their usage in research is increasing.
- For further motivation read [why Deepmind uses jax here](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research).

There are exciting developments for accelerating sparsity in neural networks (K:N sparsity, CPU-acceleration, activation sparsity) and various libraries aim to enable such acceleration (todo). *jaxpruner* focuses mainly on accelerating algorithms research for sparsity. We mock sparsity by using binary masks and use dense operations for simulating sparsity. In the longer run, we also plan to provide integration with the [jax.experimental.sparse](https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html), aim to reduce the memory footprint of our models.

*jaxpruner* has 3 tenets: 
- **Fast Integration**: requires minimal changes to use.
- **Research First**: provides strong baselines and is easy to modify.
- **Minimal Overhead**: runs as fast as (dense) baseline.

### Fast Integration
Research in Machine Learning is fast paced.  This and the huge variety of Machine Learning applications result in a high number of ever-changing codebases. At the same time,adaptability of new research ideas highly correlates with their ease of use.  Therefore, JaxPruner is designed to be easily integrated into existing codebases with minimal changes.JaxPruner uses the popular [optax](https://github.com/deepmind/optax) optimization library to achieve this, requiring minimal changes when integrating with existing libraries.  State variables (i.e.  masks, counters) needed for pruning and sparse training algorithms are stored together with the optimization state, which makes parallelization and checkpointing easy.

```python
tx, params = _existing_code()
pruner = jaxpruner.MagnitudePruning(**config) # Line 1: Create pruner.
tx = pruner.wrap_optax(tx) # Line 2: Wrap optimizer.
```

### Research First
Often research projects require running multiple algorithms and baselines and so they benefit greatly from rapid prototyping. JaxPruner achieves this by committing to a generic API shared among different algorithms, which in return makes it easy to switch between different algorithms. We provide implementations for common baselines and make them easy to modify. A quick overview of such features are discussed in our colabs.

### Minimal Overhead
Sparse training and various pruning recipes requires some additional operations like masking. When we implement such basic operations we aim to minimize the overhead introduced (both memory and compute) and be as fast as the dense baseline.

```python
pruner = jaxpruner.MagnitudePruning(is_packed=True) # Line 1: Reduces mask overhead.
```
## Installation
First clone this repo.

``
git clone https://github.com/google-research/jaxpruner.git
cd jaxpruner
```

Following script creates a virtual environment and installs the necessary libraries. Finally, it runs the tests.

```
bash run.sh
```

## Quickstart
See our Quickstart colab (coming soon).

## Baselines
We will share our baseline experiments here shortly.

## Citation
Work in Progress

## Disclaimer
This is not an officially supported Google product.
