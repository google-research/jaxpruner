# Copyright 2023 The Jaxpruner Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

__version__ = 0.1

JAX_URL = "https://storage.googleapis.com/jax-releases/jax_releases.html"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jaxpruner",
    version=__version__,
    author="Google",
    author_email="jaxpruner-dev@google.com",
    description="JaxPruner: A concise library for sparsity research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/jaxpruner",
    license="Apache 2.0",
    packages = find_packages(
        where='jaxpruner',
        exclude=['t5x*', 'dopamine*', 'scenic', 'fedjax']
    ),
    zip_safe=False,
    install_requires=[
      "chex",
      "flax,
      "jax",
      "jaxlib",
      "optax",
      "numpy",
      "ml-collections"
    ],
    dependency_links=[JAX_URL],
    python_requires=">=3.6",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
