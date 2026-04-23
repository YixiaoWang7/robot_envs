# generate the eval results.

python tests/test_configs/generate_eval_task_sets.py --overwrite --out_dir tests/test_configs/1s



python tests/eval_policy_l4.py --config tests/test_configs/1s/L_mid.json --checkpoint /mnt/ssd1/yixiao/cg/ckpt/l4_1s_L_mid_transformer_cross_seed0/checkpoints/ckpt_final.pt --n_trials_per_task 25 ; python tests/eval_policy_l4.py --config tests/test_configs/1s/diag_mid.json --checkpoint /mnt/ssd1/yixiao/cg/ckpt/l4_1s_diag_mid_transformer_cross_seed0/checkpoints/ckpt_final.pt --n_trials_per_task 25 ; python tests/eval_policy_l4.py --config tests/test_configs/1s/diag_cor.json --checkpoint /mnt/ssd1/yixiao/cg/ckpt/l4_1s_diag_cor_transformer_cross_seed0/checkpoints/ckpt_final.pt --n_trials_per_task 25 ; python tests/eval_policy_l4.py --config tests/test_configs/1s/diag.json --checkpoint /mnt/ssd1/yixiao/cg/ckpt/l4_1s_diag_transformer_cross_seed0/checkpoints/ckpt_final.pt --n_trials_per_task 25 



python tests/eval_policy_l4.py --config tests/test_configs/1s/L_mid.json --checkpoint /mnt/ssd1/yixiao/cg/ckpt/l4_1s_L_mid_transformer_cross_seed0/checkpoints/ckpt_final.pt --n_trials_per_task 1