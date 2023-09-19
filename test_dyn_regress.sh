# diff_path="logs/pickplace/diffusion/timestamp_23-0915-004443&0_keep/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0916-014735&0/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0916-030832&0/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0916-110230&0_w1.5/model/policy.pth"
# diff_path="logs/pickplace_easy/onlinercsl/rcsl_diff/timestamp_23-0917-052910&0/model/policy_best.pth"
# diff_path="logs/pickplace_easy/onlinercsl/rcsl_diff/timestamp_23-0917-073935&0/checkpoint/policy_best.pth"
diff_path="logs/pickplace_easy/mbrcsl_regress&eval_episodes=50&task_weight=1.5/diffusion/timestamp_23-0918-103814&0_keep/model/policy.pth"
# diff_path=None
# dyn_path="logs/pickplace/test_dyn/dynamics/timestamp_23-0915-015639-s0_keep/model"
# dyn_path="logs/pickplace_easy/mbrcsl_regress/dynamics/timestamp_23-0918-074944&0_keep/model"
# dyn_path="logs/pickplace_easy/test_dyn&eval_episodes=10&task_weight=1.5/dynamics_regress/timestamp_23-0919-004449&0_keep/model"
dyn_path=None
# rollout_path="logs/pickplace/test_dyn/rollout/timestamp_23-0915-061633&0_1000/checkpoint"
# rollout_path="logs/pickplace/test_dyn/rollout/timestamp_23-0915-054333&0_10000_sparse/checkpoint"
# rollout_path="logs/pickplace/test_dyn/rollout_true/timestamp_23-0915-211431&0_keep/checkpoint"
rollout_path="logs/pickplace_easy/test_dyn/rollout_true/timestamp_23-0916-222425&0_data/checkpoint"
# rollout_path=None

# for iter in 5
# do
# python run_example/pickplace/test_dynamics_pickplace.py \
#     --behavior_epoch 100 --num_diffusion_iters $iter --horizon 40 \
#     --load_diffusion_path ${diff_path} \
#     --load-dynamics-path ${dyn_path} \
#     --rollout_epochs 250
# done

# python run_example/pickplace/run_mbrcsl_pickplace.py \
#     --behavior_epoch 30 --num_diffusion_iters 10 --horizon 40 \
#     --load_diffusion_path ${diff_path} \
#     --load-dynamics-path ${dyn_path} \
#     --rollout_ckpt_path ${rollout_path}

# python run_example/pickplace/run_onlinercsl_pickplace.py \
#     --task pickplace_easy \
#     --behavior_epoch 30 --num_diffusion_iters 5 --horizon 40 \
#     --load_diffusion_path ${diff_path} \
#     --load-dynamics-path ${dyn_path} \
#     --rollout_ckpt_path ${rollout_path} --rollout-batch 1 --rollout_epochs 10000 --num_need_traj 40 \
#     --eval_episodes 50 --rcsl-epoch 100 \
#     --num_workers 4

python run_example/pickplace/test_dynamics_regress_pickplace.py \
    --task 'pickplace_easy' --horizon 40 \
    --load-dynamics-path ${dyn_path} \
    --load_diffusion_path ${diff_path} --num_diffusion_iters 5\
    --rollout_epochs 50
