# Intro
This repository adopts the Idea of [Kipf et al.](https://arxiv.org/abs/1802.04687) for power-systems.
We try to detect latent edges that are useful for timeseries predictions of a developing powergrid.
# Setup
I used `conda 24.9.1` and `python 3.12.10`

## Dependencies
### Step 0: (Optional) If you are on bwUniCluster:
```commandline
module load devel/miniforge
```
### Step 1: Install dependencies via pip
**If you use a virtual env**
```commandline
pip install -r requirements.txt
```
**If you use conda**
```commandline
conda env create -f environment.yaml
conda activate RL
```
> [!WARNING]  
> This step might fail because l2rpn-baselines which we will install from source is included in the environment.yml file.
> If this is problematic just remove the corresponding line and try again.
### Step 2: Install l2rpn-baselines including submodules from source:
Uninstall previously installed versions of `l2rpn-baselines`
```bash
pip uninstall l2rpn-baselines
```
and install it from source including submodules
```bash
git clone --recurse-submodules git@github.com:Grid2op/l2rpn-baselines.git # or http: https://github.com/rte-france/l2rpn-baselines.git
cd l2rpn-baselines
pip3 install -U .
cd ..
rm -rf l2rpn-baselines
```
## Download and split scenarios
Download the required data. This may take a while.
```commandline
python setup_envs.py
```
Now your home directory (under linux, for other os I don't know) will contain the powergrid data used in the episodes.
Furthermore, it is split into training, testing, and validation episodes.
# Project
This project contains the following packages:
- **baselines**: Trains and evaluates baseline agents
- **common**: Code that is needed by various packages
- **hydra_configs**: I use hydra to inject experiment parameters. The configs containing these parameters are in this package.
- **nri**: Implements the latent edge discovery as well as an GNN-agent on these hidden edges
- **test**: Unittests
- **visualization**: Notebooks to create figures

Running anything in this project should create output in the `data`-folder. This data can then be visualized by the 
visualization notebook.



