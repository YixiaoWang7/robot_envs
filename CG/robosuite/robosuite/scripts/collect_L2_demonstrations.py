"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from collect_human_demonstrations import collect_human_trajectory, gather_demonstrations_as_hdf5

import random

object_A_names = ["cross", "cube", "cylinder"]
object_B_names = ["bin", "cup", "plate"]

def instructions_constructor(object_A_idx, object_B_idx):
    return "place the {} into the {}".format(object_A_names[object_A_idx], object_B_names[object_B_idx])

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="CG_L2")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mujoco",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    parser.add_argument(
        "--demo_strategy",
        type=str,
        default="L",
        help="Which demonstration strategy to use. Options include 'L' (L-shape), 'S' (Stairs)",
    )
    parser.add_argument(
        "--demo_length",
        type=int,
        default=20,
        help="How many demonstrations to collect in each composition",
    )
    parser.add_argument(
        "--object_indices",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # base_instruction = [0, 0]
    # instruction_idx_list = [base_instruction.copy()]
    # if args.demo_strategy == "L":
    #     for dim in range(2):
    #         current_instruction = base_instruction.copy()
    #         for idx in range(1, 3):
    #             current_instruction[dim] = idx
    #             instruction_idx_list.append(current_instruction.copy())
    #             print(current_instruction)
    # elif args.demo_strategy == "S":
    #     current_instruction = base_instruction.copy()
    #     for idx in range(1, 3):
    #         for dim in range(2):
    #             current_instruction[dim] = idx
    #             instruction_idx_list.append(current_instruction.copy())

    # instruction_idx_list = [[0, 0], [1, 0], [2, 0],
    #                         [0, 1], [1, 1], [2, 1],
    #                         [0, 2],         [2, 2]]
    # instruction_idx_list = [[2, 0]]
    if args.object_indices is not None:
        instruction_idx_list = [[int(args.object_indices[0]), int(args.object_indices[1])]]
            

    # collect demonstrations
    for instruction_idx in instruction_idx_list:
        object_A_idx, object_B_idx = instruction_idx
        instruction = instructions_constructor(object_A_idx, object_B_idx)
        idx_string = str(object_A_idx) + str(object_B_idx)
        new_dir_w_instruction = os.path.join(new_dir, idx_string)
        os.makedirs(new_dir_w_instruction)
        env.update_instruction(instruction)
        success_count = 0
        
        while success_count < args.demo_length:
            print("Current Instruction: {}".format(instruction))
            print("Collecting demonstration {}/{}".format(success_count + 1, args.demo_length))
            collect_human_trajectory(env, device, args.arm, args.max_fr)
            success_count = gather_demonstrations_as_hdf5(tmp_directory, new_dir_w_instruction, env_info)
