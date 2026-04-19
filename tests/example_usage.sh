#!/bin/bash
# Example usage of eval_policy_l4.py with config files

# Example 1: Full L4 evaluation with all 16 tasks
# Uncomment and update the checkpoint path
# python eval_policy_l4.py --config eval_config.json

# Example 2: Single task evaluation (quick testing)
# python eval_policy_l4.py --config eval_config_single_task.json

# Example 3: Use config but override some parameters
# python eval_policy_l4.py --config eval_config.json --n-episodes 100 --seed 42

# Example 4: Legacy command line mode (no config file)
# python eval_policy_l4.py \
#     --checkpoint /path/to/checkpoint.pt \
#     --device cuda \
#     --n-episodes 50 \
#     --num-envs 2 \
#     --horizon 200 \
#     --n-execute 8 \
#     --max-videos 10 \
#     --seed 0 \
#     --out-dir artifacts

echo "Edit this script and uncomment the desired command"
echo "See EVAL_CONFIG_README.md for detailed documentation"
