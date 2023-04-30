"""Jaxpruner setup configuration."""
from setuptools import find_packages
from setuptools import setup

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
    packages=find_packages(
        exclude=["*test.py", "algorithms/*.py"],
    ),
    zip_safe=False,
    install_requires=[
        "chex",
        "flax",
        "jax",
        "jaxlib",
        "optax",
        "numpy",
        "ml-collections",
    ],
    dependency_links=[JAX_URL],
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
