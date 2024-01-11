# 1 Conda Virtual Environments

If you use conda for managing Python virtual environments, first you need to install either [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

## 1.1 Libmamba Solver

Conda's own solver is very slow, so I recommend using `Libmamba`. To use the new solver, first update conda in the base environment (optional step):

```bash
conda update -n base conda
```

Then install and activate `Libmamba` as the solver:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## 1.2 Environments

Following YAML configuration files for Conda environments are available:

* [3d-computer-vision.yml](./3d-computer-vision.yml)
  * **Environment name:** 3d-computer-vision
  * **Contains:** pykitti, open3d, opencv, numpy etc.

You can create a new virtual environment as follows:

```bash
conda env create -f <NAME-OF-THE-FILE>
```

Once the environment has been created, you can activate it by executing the following command:

```bash
conda activate <NAME-OF-THE-ENVIRONMENT>
```

