# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import NoetixBumi.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

GRAVEL_TERRAINS_CFG_V1 = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.6, noise_range=(0.01, 0.05), noise_step=0.01),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.2, slope_range=(0.05, 0.2)),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=0.2, slope_range=(0.05, 0.2)),
    },
)

GRAVEL_TERRAINS_CFG_V2 = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.3, noise_range=(0.01, 0.1), noise_step=0.02),
        "DiscreteObstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.1,
            border_width=0.5,
            obstacle_width_range=(0.5, 1),
            obstacle_height_range=(0.05, 0.2),
            num_obstacles=50,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.1, slope_range=(0.1, 0.5)),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=0.1, slope_range=(0.1, 0.5)),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.20), step_width=0.3
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.20), step_width=0.3
        ),
        "stairs_up_narrow": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.20), step_width=0.2
        ),
        "stairs_down_narrow": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.20), step_width=0.2
        ),
    },
)

GRAVEL_TERRAINS_CFG_V3 = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.5, num_waves=10.0, amplitude_range=(0.01, 0.04)),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.01, 0.05), noise_step=0.015
        ),
    }
)

GRAVEL_TERRAINS_CFG_V4 = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.4, noise_range=(0.0, 0.0), noise_step=0.02),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.3, num_waves=10.0, amplitude_range=(0.02, 0.05)),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.01, 0.03), noise_step=0.01
        ),
    }
)

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.3, noise_range=(0.01, 0.1), noise_step=0.02),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.1, obstacle_width_range=(0.2, 1.0), obstacle_height_range=(0.05, 0.1), num_obstacles=10
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.15, slope_range=(0.1, 0.5)),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=0.15, slope_range=(0.1, 0.5)),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.05, 0.2), step_width=0.3
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.05, 0.2), step_width=0.3
        ),
    },
)

BUMI1_WALKRUN_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.6, noise_range=(0.0, 0.0), noise_step=0.02),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.3, noise_range=(0.01, 0.05), noise_step=0.02),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.15, slope_range=(0.05, 0.2)),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=0.15, slope_range=(0.05, 0.2)),
    },
)

MIMIC_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.5, num_waves=10.0, amplitude_range=(0.02, 0.05)),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.01, 0.05), noise_step=0.015
        ),
    },
)

