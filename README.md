# Template for Noetix-Bumi Isaac Lab Projects

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RK](https://img.shields.io/badge/RSL_RL-3.1.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- ***Note that the Isaacsim version must be 5.0.0!!!***
- We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/NoetixBumi
    python -m pip install -e rsl_rl
- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/rsl_rl/train.py --task=<TASK_NAME>
        
        For this exmple:
        python scripts/rsl_rl/train.py --task=NoHim-21DOF-WalkRun-AMP-v0 --headless --num_envs=4096
        ```

## Usage

### Visualize motion

Visualize the motion by updating the simulation with data from Bumi/datasets/motion_visualization.

```bash
# WalkRun
python scripts/rsl_rl/play_animation.py --task=NoHim-21DOF-WalkRun-AMP-v0 
#SPACE pause/resume, N next, B previous, R restart, Q quit.
# mimic
python scripts/motion/replay_npz_bumi.py --motion_file <motion_path>
#For this Motion file: 
python scripts/motion/replay_npz_bumi.py --motion_file source/NoetixBumi/NoetixBumi/assets/datasets/mimic_data/bumi/2.npz
```


### Train

Train the policy using AMP expert data from Bumi/datasets/motion_amp_expert.

```bash
# WalkRun
python scripts/rsl_rl/train.py --task=NoHim-21DOF-WalkRun-AMP-v0 --headless --num_envs=4096
#muti GPU:CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train.py --task=NoHim-21DOF-WalkRun-AMP-v0 --headless --num_envs=4096 

# Mimic
python scripts/rsl_rl/train.py --task=Mimic-Bumi-v0 --headless --num_envs=4096

### Play

Run the trained policy.

```bash
# WalkRun
python scripts/rsl_rl/play.py --task=NoHim-21DOF-WalkRun-AMP-v0 --num_envs=1 
#  --video --video_length=1000
# Mimic
python scripts/rsl_rl/play.py --task=Mimic-Bumi-v0 --num_envs=1
```
 
### Sim2Sim(MuJoCo)

Evaluate the trained policy in MuJoCo to perform cross-simulation validation.

Exported_policy/ contains pretrained policies provided by the project. When using the play script, trained policy is exported automatically and saved to path like logs/run/[timestamp]/exported/policy.pt.
```bash
# WalkRun
python scripts/mujoco/sim2sim_walkrun_bumi.py --policy Exported_policy/walk.onnx --duration 100
# Mimic
python scripts/mujoco/sim2sim_mimic_bumi.py --policy Exported_policy/mimic.onnx --motion_file <motion_path> --duration 100
```

### Export Json
Transformer npz file to Json file, for deploy. Modify `motion_file` and `out_dir`
```bash
python scripts/motion/npz_to_deplay_json.py
```

### Tensorboard
```bash
tensorboard --logdir=logs
```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/NoetixBumi"
        "<path-to-rsl-rl>/rsl_rl",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_tasks",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_mimic",
        "<path-to-isaac-lab>/IsaacLab/source/extensions",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_assets",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab_rl",
        "<path-to-isaac-lab>/IsaacLab/source/isaaclab",
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```# noetix_bumi_lab
