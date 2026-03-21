import h5py
import numpy as np
import argparse
import os
import sys

'''
This script shows some attributes of the dataset.
'''

def show(input_file):
        
    print(f"Processing {input_file}...")
    with h5py.File(input_file, 'r') as h5_in:
        if 'data' not in h5_in:
            raise ValueError("No 'data' group found in the input file.")
            
        for demo_key in h5_in['data']:
            demo = h5_in['data'][demo_key]
            # breakpoint()
            # demo.keys()
            # > <KeysViewHDF5 ['actions', 'datagen_info', 'obs', 'src_demo_inds', 'src_demo_labels', 'states']>
            obs = demo['obs']
            # breakpoint()
            # obs.keys()
            # > <KeysViewHDF5 ['agentview_image', 'object', 'robot0_eef_pos', 'robot0_eef_quat',
            #                  'robot0_eef_quat_site', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos',
            #                  'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos',
            #                  'robot0_joint_pos_sin', 'robot0_joint_vel']>
            agentview_image = obs['agentview_image']
            import matplotlib.pyplot as plt # you may also use cv2
            plt.imshow(agentview_image[0])
            plt.title("Agentview Image (First Frame)")
            plt.show()

def modify_hdf5_data(dataset_name, args):
    with h5py.File(dataset_name, 'a') as f:
        demos = list(f["data"].keys())
        
        for ep in demos:
            actions = np.array(f[f"data/{ep}/actions"][()])
            print("Original action:", actions[0])
            
            if args.x is not None:
                actions[:, 0:6] *= args.x
            if args.c is not None:
                actions[:, 0:6] = np.clip(actions[:, 0:6], -args.c, args.c)
            if args.g is not None:
                actions[:, 6] *= args.g
            if args.x is None and args.c is None and args.g is None:
                raise ValueError("Please provide at least one of the --x, --c and --g argument to modify actions.")
            print("Modified action:", actions[0])
            
            if 'data' not in f:
                f.create_group('data')
            if ep not in f['data']:
                f['data'].create_group(ep)
            
            if f"data/{ep}/actions" in f:
                del f[f"data/{ep}/actions"]
            f.create_dataset(f"data/{ep}/actions", data=actions)

if __name__ == "__main__":
    input_file = 'tasks/demo_00.hdf5'

    show(input_file)
    print("done")