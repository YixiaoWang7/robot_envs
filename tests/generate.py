#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import json
import logging
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Callable

import einops
import gymnasium as gym
import h5py
import numpy as np
import torch
from termcolor import colored
from torch import nn
from tqdm import tqdm, trange

from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import add_envs_task, check_env_attributes_and_types, preprocess_observation_cg
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    disable_progbar: bool = False,
    steps_after_success: int = 5,  # Cut off N steps after success
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    # Track when each environment first succeeded (-1 means not yet)
    success_step = np.array([-1] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm() or disable_progbar,  # Disable if in slurm or explicitly requested
        leave=False,
    )
    check_env_attributes_and_types(env)
    policy_obs_filter = ['task','observation.state','observation.environment_state']
    while not np.all(done):
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation_cg(observation)

        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)
        

        # print(f"task: {observation['task']}")

        with torch.inference_mode():
            inference_start_time = time.time()
            action = policy.select_action({k: observation[k] for k in policy_obs_filter})
            inference_duration = time.time() - inference_start_time
            # if inference_duration > 0.005:
            #     print(f"Step {step}: Policy inference time: {inference_duration:.4f} seconds")

        # Convert to CPU / numpy.
        action = action.to("cpu").numpy()
        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available of none of the envs finished.
            
        successes = info["is_success"]
        
        # Track when each environment first succeeds
        for env_idx in range(env.num_envs):
            if successes[env_idx] and success_step[env_idx] == -1:
                success_step[env_idx] = step

        # Keep track of which environments are done so far.
        # Also mark as done if we've reached steps_after_success after success
        for env_idx in range(env.num_envs):
            if success_step[env_idx] != -1 and step >= success_step[env_idx] + steps_after_success:
                done[env_idx] = True
        
        done = terminated | truncated | done

        all_actions.append(torch.from_numpy(action))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation_cg(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret["observation"] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def save_successful_episodes_to_hdf5(
    hdf5_path: Path,
    episode_data: dict,
    episode_id: int,
) -> None:
    """Save a successful episode to HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        episode_data: Dictionary containing episode data
        episode_id: Episode identifier
    """
    with h5py.File(hdf5_path, "a") as f:
        # Create demo group for this episode
        demo_name = f"demo_{episode_id}"
        if demo_name in f:
            del f[demo_name]  # Remove if exists
        
        demo_grp = f.create_group(demo_name)
        
        # Save observations (states)
        obs_grp = demo_grp.create_group("obs")
        if "states" in episode_data:
            obs_grp.create_dataset("states", data=episode_data["states"], compression="gzip")

        if "environment_state" in episode_data:
            obs_grp.create_dataset("environment_state", data=episode_data["environment_state"], compression="gzip")
        
        # Save images (using consistent naming convention)
        if "agentview_image" in episode_data:
            # print("agentview_image shape:", episode_data["agentview_image"].shape)
            obs_grp.create_dataset(
                "agentview_image", 
                data=episode_data["agentview_image"],
                compression="gzip"
            )
 
        if "robot0_eye_in_hand_image" in episode_data:
            # print("robot0_eye_in_hand_image shape:", episode_data["robot0_eye_in_hand_image"].shape)
            obs_grp.create_dataset(
                "robot0_eye_in_hand_image",
                data=episode_data["robot0_eye_in_hand_image"],
                compression="gzip"
            )
        
        # Save actions
        if "actions" in episode_data:
            demo_grp.create_dataset("actions", data=episode_data["actions"], compression="gzip")
        
        # Save rewards
        if "rewards" in episode_data:
            demo_grp.create_dataset("rewards", data=episode_data["rewards"], compression="gzip")
        
        # Save metadata
        demo_grp.attrs["num_samples"] = len(episode_data["actions"])
        demo_grp.attrs["success"] = True
        if "task" in episode_data:
            demo_grp.attrs["task"] = episode_data["task"]


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    n_episodes: int = None,
    n_success_episodes: int = None,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    start_seed: int | None = None,
    save_hdf5: bool = False,
    hdf5_dir: Path | None = None,
    steps_after_success: int = 5,  # Cut off N steps after success
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: Total number of episodes to run (if specified, ignores n_success_episodes).
        n_success_episodes: Number of successful episodes to collect (runs until this many successes).
        max_episodes_rendered: Maximum number of episodes to render into videos (0 to disable).
        videos_dir: Where to save rendered videos.
        start_seed: The first seed to use for the first individual rollout.
        save_hdf5: Whether to save successful episodes to HDF5.
        hdf5_dir: Directory to save HDF5 files.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")
    
    if save_hdf5 and not hdf5_dir:
        raise ValueError("If save_hdf5 is True, hdf5_dir must be provided.")
    
    if n_episodes is None and n_success_episodes is None:
        raise ValueError("Either n_episodes or n_success_episodes must be specified.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # Determine mode: success-based or episode-based
    use_success_counter = (n_success_episodes is not None)
    target_successes = n_success_episodes if use_success_counter else None

    # Keep track of metrics
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    n_episodes_saved = 0  # for HDF5 saving
    n_total_episodes = 0  # total episodes run
    
    # Prepare HDF5 file
    if save_hdf5:
        hdf5_dir.mkdir(parents=True, exist_ok=True)
        hdf5_path = hdf5_dir / "successful_episodes.hdf5"
        logging.info(f"Will save successful episodes to: {hdf5_path}")
        
    # Progress bar for success collection mode
    if use_success_counter:
        progbar = tqdm(total=target_successes, desc=f"Collecting {target_successes} successful episodes")
    else:
        n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)
        progbar = tqdm(total=n_batches, desc=f"Running {n_episodes} episodes")

    # Main loop: keep running until target is met
    batch_ix = 0
    while True:
        # Check stopping condition
        if use_success_counter:
            if n_episodes_saved >= target_successes:
                break
        else:
            if n_total_episodes >= n_episodes:
                break
        
        # Set seeds for this batch
        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        # Run rollout (disable inner progress bar when using success counter)
        rollout_data = rollout(
            env,
            policy,
            seeds=list(seeds) if seeds else None,
            return_observations=save_hdf5,  # Only return observations if saving to HDF5
            render_callback=None,  # Disable rendering for speed
            disable_progbar=use_success_counter,  # Disable inner progress bar when collecting successes
            steps_after_success=steps_after_success,  # Cut off N steps after success
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        batch_successes = batch_successes.tolist()
        all_successes.extend(batch_successes)
        n_total_episodes += len(batch_successes)
        
        # Save successful episodes to HDF5
        if save_hdf5:
            for env_idx in range(env.num_envs):
                if batch_successes[env_idx]:  # Only save successful episodes
                    done_idx = done_indices[env_idx].item()
                    
                    # Extract episode data
                    episode_dict = {
                        "actions": rollout_data["action"][env_idx, :done_idx+1].cpu().numpy(),
                        "rewards": rollout_data["reward"][env_idx, :done_idx+1].cpu().numpy(),
                        "task": env.unwrapped.envs[env_idx].task,
                    }
                    
                    # Extract observations (states and images)
                    if "observation" in rollout_data:
                        obs = rollout_data["observation"]
                        
                        # States
                        if "observation.state" in obs:
                            episode_dict["states"] = obs["observation.state"][env_idx, :done_idx+2].cpu().numpy()

                        # Environment state
                        if "observation.environment_state" in obs:
                            episode_dict["environment_state"] = obs["observation.environment_state"][env_idx, :done_idx+2].cpu().numpy()
                        
                        # Agent view image (matches wrapper key: observation.images.agentview)
                        if "observation.images.agentview" in obs:
                            episode_dict["agentview_image"] = obs["observation.images.agentview"][env_idx, :done_idx+2].cpu().numpy()
                        
                        # Eye-in-hand image (matches wrapper key: observation.images.robot0_eye_in_hand)
                        if "observation.images.robot0_eye_in_hand" in obs:
                            episode_dict["robot0_eye_in_hand_image"] = obs["observation.images.robot0_eye_in_hand"][env_idx, :done_idx+2].cpu().numpy()
                    
                    # Save to HDF5
                    save_successful_episodes_to_hdf5(hdf5_path, episode_dict, n_episodes_saved)
                    n_episodes_saved += 1
                    
                    # Update progress bar
                    if use_success_counter:
                        progbar.update(1)
                        progbar.set_postfix({
                            "total_episodes": n_total_episodes,
                            "success_rate": f"{(n_episodes_saved/n_total_episodes)*100:.1f}%"
                        })
        
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)
        
        # Update progress bar for episode-based mode
        if not use_success_counter:
            progbar.update(1)
            progbar.set_postfix({
                "success_rate": f"{np.mean(all_successes)*100:.1f}%",
                "n_success": sum(all_successes)
            })
        
        batch_ix += 1

    progbar.close()
    
    # Compile eval info
    total_time = time.time() - start
    info = {
        "aggregated": {
            "n_total_episodes": n_total_episodes,
            "n_successful_episodes": sum(all_successes),
            "n_failed_episodes": n_total_episodes - sum(all_successes),
            "success_rate": float(np.mean(all_successes) * 100),
            "avg_sum_reward": float(np.nanmean(sum_rewards)),
            "avg_max_reward": float(np.nanmean(max_rewards)),
            "eval_s": total_time,
            "eval_ep_s": total_time / n_total_episodes if n_total_episodes > 0 else 0,
        },
    }
    
    if save_hdf5:
        info["aggregated"]["n_episodes_saved_hdf5"] = n_episodes_saved
        info["aggregated"]["hdf5_path"] = str(hdf5_path)

    return info


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )


    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy(
            env,
            policy,
            n_success_episodes=500,  # Collect 500 successful episodes
            max_episodes_rendered=0,  # Disable video rendering for speed
            start_seed=cfg.seed,
            save_hdf5=True,
            hdf5_dir=Path(cfg.output_dir) / "hdf5_data",
            steps_after_success=5,  # Cut off 5 steps after success
        )
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(info["aggregated"])
    print("="*80)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    env.close()

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
