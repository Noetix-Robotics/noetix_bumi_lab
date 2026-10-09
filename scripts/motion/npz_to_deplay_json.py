# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import json
import os
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[2] / "source" / "NoetixBumi" / "NoetixBumi" / "assets"

# 输入文件路径
motion_file = ASSET_DIR / "datasets/mimic_data/bumi/0-260126_216_dance_zoo_001.npz"

# 输出文件路径
out_dir = ASSET_DIR / "datasets/mimic_data/bumi/json/0-260126_216_dance_zoo_001.json"

# 确保输出目录存在
os.makedirs(os.path.dirname(out_dir), exist_ok=True)

try:
    # 加载运动数据
    motion = np.load(motion_file)

    # 提取关节位置和速度数据
    motion_joint_pos = motion["joint_pos"]
    motion_joint_vel = motion["joint_vel"]
    #     body_quat_w = self.motion["body_quat_w"]
    motion_quat = motion["body_quat_w"][:, 0, :]
    #     motion_quat = motion["body_quat_w"][:, 0]
    # 将特定关节的位置和速度置零（索引14,15,18,19）
    # motion_joint_pos[:, [14, 15, 18, 19]] = 0.0
    # motion_joint_vel[:, [14, 15, 18, 19]] = 0.0

    # 构建数据字典（修复了语法错误，添加了冒号）
    data = {
        "metadata": {
            "original_file": str(motion_file),
            # "processed_joints": [14, 15, 18, 19],
            "frames_count": len(motion_joint_pos),
            "joints_count": motion_joint_pos.shape[1] if motion_joint_pos.ndim > 1 else 1,
        },
        "joint_pos": motion_joint_pos.tolist(),
        "joint_vel": motion_joint_vel.tolist(),
        "body_quat_w": motion_quat.tolist(),
    }

    # 保存为JSON文件（修复了文件路径和模式）
    with open(out_dir, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"数据已成功保存到: {out_dir}")
    print(f"数据统计: {len(motion_joint_pos)} 帧, {motion_joint_pos.shape[1]} 个关节")

except FileNotFoundError:
    print(f"错误: 文件未找到 - {motion_file}")
except KeyError as e:
    print(f"错误: NPZ文件中缺少必要的键 - {e}")
except Exception as e:
    print(f"错误: {e}")

# 验证保存的文件（可选）
try:
    with open(out_dir, encoding="utf-8") as f:
        loaded_data = json.load(f)
    print("文件验证成功，可以正常读取")

    # 显示一些基本信息
    if "metadata" in loaded_data:
        print(f"验证信息: {loaded_data['metadata']}")

except Exception as e:
    print(f"文件验证失败: {e}")
