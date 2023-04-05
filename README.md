# JaxPruner: a research library for sparsity research

## Introduction
Sparsity is an active research area for achieving better efficiency in Deep Learning. However, utilizing sparsity and realizing its potential requires a closer collaboration between hardware, software and algorithms research. Such collaborations often require a flexible library which enables rapid prototyping and evaluation on a variety of benchmarks.

We believe a sparsity library in Jax has the potential to greatly accelerate sparsity research. This is because:

- Over the last few years Jax became the leading framework for research in Google and many of the state-of-the-art models use jax libraries like [scenic](https://github.com/google-research/scenic) and [t5x](https://github.com/google-research/t5x).

- Functional nature of jax enables quick integration where models or parameters can be transformed, contrast to changing the source code (model definition, etc.).

- Also due its functional nature, parameters outside of the model definition. This makes parameter updates and masking weights incredibly simple

- For further motivation read [why Deepmind uses jax here](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research).

There are exciting developments for accelerating sparsity in neural networks (K:N sparsity, CPU-acceleration, activation sparsity) and various libraries aim to enable such acceleration (todo). *jaxpruner* focuses mainly on accelerating algorithms research for sparsity. We mock sparsity by using binary masks and use dense operations for simulating sparsity. In the longer run, we also plan to provide integration with the [jax.experimental.sparse](https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html), aim to reduce the memory footprint of our models.

*jaxpruner* has 3 tenets: 
- **Fast Integration**: requires minimal changes to use.
- **Easy Research**: provides strong baselines and is easy to modify.
- **Minimal Overhead**: runs as fast as (dense) baseline.

### Fast Integration
We integrate *jaxpruner* with [optax](https://github.com/deepmind/optax) optimization 
library and require no changes in model definition.

TODO add an example

### Easy Research
We provide common utilities like schedule functions, sparsity distributions and sparsity types.

TODO add an example

### Minimal Overhead
Sparse training and various pruning recipes requires some additional operations like masking. When we implement such basic operations we aim to minimize the overhead introduced (both memory and compute) and be as fast as the dense baseline.

TODO add runtime numbers.

## Installation
TODO

## Quickstart
TODO add link for the baseline

## Baselines
TODO add ViT pruning baselines.

## Citation
work in progress

