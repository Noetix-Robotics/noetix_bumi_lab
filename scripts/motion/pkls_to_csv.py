import joblib
import torch

start_idx = 0
end_idx = -1

def process_rawframes(data, dt, start_idx, end_idx):
    root_pos = torch.tensor(data['root_trans_offset'], dtype=torch.double, device='cpu', requires_grad=False)[start_idx:end_idx]
    root_rot = torch.tensor(data['root_rot'], dtype=torch.double, device='cpu', requires_grad=False)[start_idx:end_idx]
    dof_pos = torch.tensor(data['dof'], dtype=torch.double, device='cpu', requires_grad=False)[start_idx:end_idx]   
    # dof_pos[:, [5, 11]] = 0.0
    # root_pos[:, 2] += 0.1
    # dof_pos[:, 15] += 0.2
    # dof_pos[:, 20] += -0.2

    frames_tensor = torch.cat((root_pos[1:], root_rot[1:], dof_pos[1:]), dim=1)

    
    return frames_tensor.tolist()

if __name__ == "__main__":
    from pathlib import Path
    # 定义输入输出路径
    MOTION_FILE_Dir = "/home/zsp/project/noetix_bumi_lab/source/NoetixBumi/NoetixBumi/assets/datasets/noetix_bumi_2.pkl"  
    OUTPUT_FILE_Dir = "/home/zsp/project/noetix_bumi_lab/source/NoetixBumi/NoetixBumi/assets/datasets/noetix_bumi_2.csv"
    input_path = Path(MOTION_FILE_Dir)
    output_path = Path(OUTPUT_FILE_Dir)

    # 支持单个.pkl文件和目录批量转换
    if input_path.is_file():
        pkl_files = [input_path]
    elif input_path.is_dir():
        pkl_files = sorted(input_path.rglob("*.pkl"))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in: {input_path}")

    for pkl_file in pkl_files:
        if input_path.is_file():
            csv_file = output_path
        else:
            rel = pkl_file.relative_to(input_path).with_suffix(".csv")
            csv_file = output_path / rel

        csv_file.parent.mkdir(parents=True, exist_ok=True)

        with open(pkl_file, 'rb') as file:
            motion = joblib.load(file)

        print(motion.keys())
        for k, data in motion.items():

            if len(motion) > 1:
                key_csv_file = csv_file.with_name(f"{csv_file.stem}_{k}{csv_file.suffix}")
            else:
                key_csv_file = csv_file

            frames_dt = 1. / data["fps"]
            frames = data['dof'].shape[0]
            print("Total motion frames: %d, Total motion durition: %f" %(frames, frames * frames_dt))

            # 写入CSV文件
            # 获取处理后的帧数据
            processed_data = process_rawframes(data, frames_dt, start_idx, end_idx)
            import csv
            # 保存为CSV格式
            with open(key_csv_file, "w", newline="") as f:
                # 检查processed_data的数据结构
                if isinstance(processed_data, list) and len(processed_data) > 0:
                    # 如果是字典列表，提取键作为列标题
                    if isinstance(processed_data[0], dict):
                        # 获取所有键作为列标题
                        fieldnames = processed_data[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(processed_data)
                    # 如果是列表的列表
                    elif isinstance(processed_data[0], list):
                        writer = csv.writer(f)
                        writer.writerows(processed_data)
            print("Generating " + str(key_csv_file))