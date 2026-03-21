"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
"""

import argparse
import json
import os
import random

import h5py
import numpy as np

import robosuite
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to your demonstration dataset",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    args = parser.parse_args()

    hdf5_path = args.dataset
    f = h5py.File(hdf5_path, "r")
    ff = h5py.File("/home/jix22/CG/robosuite/robosuite/models/assets/demonstrations_private/CG_L2/00/demo.hdf5", "r")
    # breakpoint()
    env_name = ff["data"].attrs["env"]
    env_info = json.loads(ff["data"].attrs["env_info"])

    env = robosuite.make(
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    for ep in demos:
        print(f"Playing {ep}")
        # read the model xml, using the metadata stored in the attribute for this episode
        # breakpoint()
        model_xml = ff["data/demo_1".format(ep)].attrs["model_file"]

        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        # if args.use_actions:

        # load the initial state
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        # load the actions and play them back open-loop
        actions = np.array(f["data/{}/actions".format(ep)][()])
        print(actions[0])
        num_actions = actions.shape[0]

        for j, action in enumerate(actions):
            obs_step = env.step(action)
            print('')
            print(["{:.3f}".format(obs_step[0]["object-state"][i]) for i in range(0,7)])
            print(["{:.3f}".format(action[i]) for i in range(0,7)])
            print('')
            env.render()
            time.sleep(0.1)
            # for i in range(6):
            #     print(obs_step[0]["object-state"][7*i : 7*i+7])
            # print(obs_step[0]["object-state"][42:])

            if j < num_actions - 1:
                # ensure that the actions deterministically lead to the same recorded states
                state_playback = env.sim.get_state().flatten()
                if not np.all(np.equal(states[j + 1], state_playback)):
                    err = np.linalg.norm(states[j + 1] - state_playback)
                    print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

        # else:

        # force the sequence of internal mujoco states one by one
        for state in states:
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            if env.renderer == "mjviewer":
                env.viewer.update()
            env.render()
    


    f.close()
