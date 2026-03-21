import argparse
import os
import h5py
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="demo.hdf5",
        help="Path to your demonstration dataset",
    )
    parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="Scale factor for the action data",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=None,
        help="Clip factor for the action data",
    )
    parser.add_argument(
        "--g",
        type=float,
        default=None,
        help="Scale factor for the gripper data",

    )
    args = parser.parse_args()

    hdf5_path = args.dataset

    # 使用'a'模式打开文件以进行修改，而不是'w'模式（会覆盖原有文件）
    with h5py.File(hdf5_path, 'a') as f:  # 使用上下文管理器确保文件正确关闭
            
        # 遍历所有演示片段
        demos = list(f["data"].keys())
        
        for ep in demos:
            # 读取动作数据
            actions = np.array(f[f"data/{ep}/actions"][()])
            print("Original action:", actions[0])
            
            # 修改动作数据
            if args.x is not None:
                actions[:, 0:6] *= args.x
            if args.c is not None:
                actions[:, 0:6] = np.clip(actions[:, 0:6], -args.c, args.c)
            if args.g is not None:
                actions[:, 6] *= args.g
            if args.x is None and args.c is None and args.g is None:
                raise ValueError("Please provide at least one of the --x, --c and --g argument to modify actions.")
            print("Modified action:", actions[0])
            
            # 确保目标文件中存在相应的组和数据集
            if 'data' not in f:
                f.create_group('data')
            if ep not in f['data']:
                f['data'].create_group(ep)
            
            # 写入修改后的数据
            if f"data/{ep}/actions" in f:
                del f[f"data/{ep}/actions"]  # 删除原有数据集
            f.create_dataset(f"data/{ep}/actions", data=actions)