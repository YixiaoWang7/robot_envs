import h5py
import numpy as np
import argparse
import os
import sys

'''
This script merges multiple HDF5 files into a single file, adding language instructions and modifying action data.
'''

def one_hot_tokenize(language_instructions):
    one_hot_dict = {"place": 0, "the": 1, "into": 2, "cross": 3, "cube": 4, "cylinder": 5, "bin": 6, "cup": 7, "plate": 8}
    words = language_instructions.split()
    language_vector = np.zeros(len(one_hot_dict))
    for word in words:
        language_vector[one_hot_dict[word]] = 1
    return language_vector

def merge_hdf5_files(input_files, output_file, language_instructions, use_simp):
    if len(input_files) != len(language_instructions):
        raise ValueError("#input_files doesn't match #language_instructions.")
    
    with h5py.File(output_file, 'w') as h5_out:
        overall_demo_count = 0
        data_group = h5_out.create_group('data')
        
        for file_idx, (input_file, instruction) in enumerate(zip(input_files, language_instructions)):
            print(f"Processing {input_file}...")
            with h5py.File(input_file, 'r') as h5_in:
                # Only process the 'data' group from input files
                if 'data' not in h5_in:
                    continue
                    
                for demo_key in h5_in['data']:
                    # Create new demo key with index if needed
                    new_demo_key = f"demo_{overall_demo_count}"
                    print(new_demo_key)
                    overall_demo_count += 1
                    
                    # Copy the entire demo group at once
                    h5_in.copy(f'data/{demo_key}', data_group, name=new_demo_key)
                    
                    # Add language instruction to the copied group
                    dest_group = data_group[new_demo_key]
                    if 'obs' in dest_group and 'language' not in dest_group['obs']:
                        num_timesteps = dest_group['obs/object'].shape[0]
                        instruction_extended = np.tile(instruction, (num_timesteps, 1))
                        dest_group['obs'].create_dataset('language', data=instruction_extended)
                    
                    if use_simp:
                        full_object = dest_group['obs/object']
                        num_timesteps = full_object.shape[0]
                        if 'obs' in dest_group and 'object_A' not in dest_group['obs']:
                            object_A = np.zeros((num_timesteps, 3+4))
                            if instruction[3] == 1:
                                object_A = full_object[:, 0:7]
                            elif instruction[4] == 1:
                                object_A = full_object[:, 7:14]
                            elif instruction[5] == 1:
                                object_A = full_object[:, 14:21]
                            dest_group['obs'].create_dataset('object_A', data=object_A)
                        if 'obs' in dest_group and 'object_B' not in dest_group['obs']:
                            object_B = np.zeros((num_timesteps, 3+4))
                            if instruction[6] == 1:
                                object_B = full_object[:, 21:28]
                            elif instruction[7] == 1:
                                object_B = full_object[:, 28:35]
                            elif instruction[8] == 1:
                                object_B = full_object[:, 35:42]
                            dest_group['obs'].create_dataset('object_B', data=object_B)

                for attr_name, attr_value in h5_in['data'].attrs.items():
                    data_group.attrs[attr_name] = attr_value

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="Scale factor for the action data",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.9,
        help="Clip factor for the action data",
    )
    parser.add_argument(
        "--g",
        type=float,
        default=0.5,
        help="Scale factor for the gripper data",
    )
    parser.add_argument(
        "--simp",
        action="store_true",
        help="Use simplified object names",
    )
    args = parser.parse_args()

    # input_files = [
    #     'demo_00.hdf5',
    #     'demo_01.hdf5',
    #     'demo_02.hdf5',
    #     'demo_10.hdf5',
    #     'demo_11.hdf5',
    #     'demo_12.hdf5',
    #     'demo_20.hdf5',
    #     'demo_21.hdf5',
    #     'demo_22.hdf5',
    # ]
    input_files = [
        'tasks/demo_00.hdf5',
        # 'tasks/demo_01.hdf5',
        # 'tasks/demo_02.hdf5',
        # 'tasks/demo_10.hdf5',
        # 'tasks/demo_11.hdf5',
        # 'tasks/demo_12.hdf5',
        # 'tasks/demo_20.hdf5',
        # 'tasks/demo_21.hdf5',
        # 'tasks/demo_22.hdf5',
    ]
    language_instructions = [
        "place the cross into the bin",
        # "place the cross into the cup",
        # "place the cross into the plate",
        # "place the cube into the bin",
        # "place the cube into the cup",
        # "place the cube into the plate",
        # "place the cylinder into the bin",
        # "place the cylinder into the cup",
        # "place the cylinder into the plate",
    ]

    language_encoding = [one_hot_tokenize(instr) for instr in language_instructions]
    use_simp = args.simp
    
    dataset_name = f"00-merged{'-simp' if use_simp else ''}-onehot-c{args.c:.1f}-g{args.g:.1f}.hdf5"
    merge_hdf5_files(input_files, dataset_name, language_encoding, use_simp)
    modify_hdf5_data(dataset_name, args)
    print("done")