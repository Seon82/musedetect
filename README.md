# 🎶 musedetect
A music instrument detection project using deep learning, which can be used to reproduce the results in our paper "A Hierarchical Deep Learning Approach for Rare Instrument Detection".
This project was built in  collaboration with [Linkaband](https://linkaband.com/).


# Installation
This package requires `python>=3.10`.

* Clone the repository.
* Run `pip install .` from the root of the project to install the package and its dependencies.

# Getting started
Examples on how to use the package to reproduce the paper's results are given in different notebooks, which can be found under [examples](./examples/).

The package provides `tensorboard` integration. Run `tensorboard --logdir log_folder` to load the logs in your browser. The directory where logs are saved is specified by the `log_dir` argument in the `train` method.


# Contributing
To contribute code to the repository:

* Install [poetry](https://python-poetry.org/docs/#installation), our dependency management tool.
* Clone the repository.
* Install the project and its dependencies: `poetry install`.
* Set up the pre-commit hooks to ensure code quality: `pre-commit install`.
  
To add dependencies to the project, use `poetry add` (for example `poetry add numpy`). 

