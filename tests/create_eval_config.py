#!/usr/bin/env python
"""
Helper script to generate evaluation config files.

Usage:
    python create_eval_config.py --output my_config.json --checkpoint /path/to/model.pt
    python create_eval_config.py --output test.json --tasks cross,cube --containers bin,mug
"""

import argparse
import json
from pathlib import Path

# CG_L4 vocabulary
_L4_OBJECTS = ["cross", "cube", "cylinder", "milk"]
_L4_CONTAINERS = ["bin", "mug", "plate", "mug_no_handle"]


def generate_tasks(objects=None, containers=None):
    """Generate task list from object and container selections."""
    objects = objects or _L4_OBJECTS
    containers = containers or _L4_CONTAINERS
    return [f"place the {obj} into the {cont}" for obj in objects for cont in containers]


def create_config(
    checkpoint="",
    device="cuda",
    n_episodes=50,
    num_envs=2,
    horizon=200,
    n_execute=8,
    max_videos=10,
    seed=0,
    out_dir="artifacts",
    tasks=None,
    save_rollouts=True,
    save_videos=True,
    save_per_episode=True,
    save_per_task=True,
    include_action_stats=True,
    include_reward_stats=True,
):
    """Create a config dictionary."""
    if tasks is None:
        tasks = generate_tasks()
    
    return {
        "eval_config": {
            "checkpoint": checkpoint,
            "device": device,
            "n_episodes": n_episodes,
            "num_envs": num_envs,
            "horizon": horizon,
            "n_execute": n_execute,
            "max_videos": max_videos,
            "seed": seed,
            "out_dir": out_dir,
        },
        "tasks": tasks,
        "result_config": {
            "save_rollouts": save_rollouts,
            "save_videos": save_videos,
            "save_per_episode_details": save_per_episode,
            "save_per_task_breakdown": save_per_task,
            "include_action_statistics": include_action_stats,
            "include_reward_statistics": include_reward_stats,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation config JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Output
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output JSON file path")
    
    # Eval config
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to checkpoint (can be updated later)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--n-execute", type=int, default=8)
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    
    # Task selection
    parser.add_argument("--objects", type=str, default=None,
                        help="Comma-separated object names (e.g., 'cross,cube')")
    parser.add_argument("--containers", type=str, default=None,
                        help="Comma-separated container names (e.g., 'bin,mug')")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Include all 16 L4 tasks (default)")
    
    # Result config
    parser.add_argument("--no-rollouts", action="store_true",
                        help="Don't save rollout NPZ files")
    parser.add_argument("--no-videos", action="store_true",
                        help="Don't save videos")
    parser.add_argument("--no-per-episode", action="store_true",
                        help="Don't save per-episode details")
    parser.add_argument("--no-per-task", action="store_true",
                        help="Don't save per-task breakdown")
    parser.add_argument("--no-action-stats", action="store_true",
                        help="Don't include action statistics")
    parser.add_argument("--no-reward-stats", action="store_true",
                        help="Don't include reward statistics")
    
    # Presets
    parser.add_argument("--preset", type=str, choices=["full", "quick", "minimal"],
                        help="Use preset configuration (overrides other task/result options)")
    
    args = parser.parse_args()
    
    # Handle presets
    if args.preset == "full":
        # Full evaluation with all details
        objects = _L4_OBJECTS
        containers = _L4_CONTAINERS
        save_rollouts = True
        save_videos = True
        save_per_episode = True
        save_per_task = True
        include_action_stats = True
        include_reward_stats = True
        n_episodes = 50
    elif args.preset == "quick":
        # Quick testing with one task
        objects = ["cross"]
        containers = ["mug_no_handle"]
        save_rollouts = True
        save_videos = True
        save_per_episode = True
        save_per_task = True
        include_action_stats = True
        include_reward_stats = True
        n_episodes = 10
    elif args.preset == "minimal":
        # Minimal output for fast iteration
        objects = ["cross"]
        containers = ["mug"]
        save_rollouts = False
        save_videos = False
        save_per_episode = False
        save_per_task = True
        include_action_stats = False
        include_reward_stats = False
        n_episodes = 10
    else:
        # Use command line arguments
        if args.objects:
            objects = [o.strip() for o in args.objects.split(",")]
        else:
            objects = _L4_OBJECTS
        
        if args.containers:
            containers = [c.strip() for c in args.containers.split(",")]
        else:
            containers = _L4_CONTAINERS
        
        save_rollouts = not args.no_rollouts
        save_videos = not args.no_videos
        save_per_episode = not args.no_per_episode
        save_per_task = not args.no_per_task
        include_action_stats = not args.no_action_stats
        include_reward_stats = not args.no_reward_stats
        n_episodes = args.n_episodes
    
    tasks = generate_tasks(objects, containers)
    
    config = create_config(
        checkpoint=args.checkpoint,
        device=args.device,
        n_episodes=n_episodes,
        num_envs=args.num_envs,
        horizon=args.horizon,
        n_execute=args.n_execute,
        max_videos=args.max_videos,
        seed=args.seed,
        out_dir=args.out_dir,
        tasks=tasks,
        save_rollouts=save_rollouts,
        save_videos=save_videos,
        save_per_episode=save_per_episode,
        save_per_task=save_per_task,
        include_action_stats=include_action_stats,
        include_reward_stats=include_reward_stats,
    )
    
    output_path = Path(args.output)
    output_path.write_text(json.dumps(config, indent=2) + "\n")
    
    print(f"Created config file: {output_path}")
    print(f"Tasks: {len(tasks)} tasks")
    print(f"  Objects: {', '.join(objects)}")
    print(f"  Containers: {', '.join(containers)}")
    print(f"Episodes: {n_episodes}")
    print(f"Save rollouts: {save_rollouts}")
    print(f"Save videos: {save_videos}")
    print()
    print("Next steps:")
    print(f"  1. Edit {output_path} to set checkpoint path")
    print(f"  2. Run: python eval_policy_l4.py --config {output_path}")


if __name__ == "__main__":
    main()
