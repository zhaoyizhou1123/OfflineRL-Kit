# diff_path="logs/pickplace/diffusion/timestamp_23-0915-004443&0_keep/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0916-014735&0/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0916-030832&0/checkpoint/policy.pth"
# diff_path="logs/pickplace_easy/mbrcsl_regress&eval_episodes=50&task_weight=1.5&sample_ratio=0.3/diffusion/timestamp_23-0919-103444&0/model/policy.pth"
# diff_path="logs/pickplace_easy/test_dyn&sample_ratio=0.25&task_weight=1.5/diffusion/timestamp_23-0919-073653&0_keep/model/policy.pth"
# diff_path="logs/pickplace_easy/test_dyn&sample_ratio=0.5&task_weight=1.5/diffusion/timestamp_23-0919-053139&0_keep/model/policy.pth"
diff_path=None
# dyn_path="logs/pickplace/test_dyn/dynamics/timestamp_23-0915-015639&0/model"
dyn_path="logs/pickplace_easy/mbrcsl_regress&eval_episodes=50&task_weight=1.5&sample_ratio=0.3/dynamics/timestamp_23-0919-103434&0/model"
# dyn_path=None
# rollout_path="logs/pickplace/test_dyn/rollout/timestamp_23-0915-061633&0_1000/checkpoint"
# rollout_path="logs/pickplace/test_dyn/rollout/timestamp_23-0915-054333&0_10000_sparse/checkpoint"
# rollout_path="logs/pickplace/test_dyn/rollout_true/timestamp_23-0915-211431&0/checkpoint"
rollout_path=None

iter=5
# for sample in 0.5 0.25 0.75
# do

# for ratio in 0.2 0.4 0.6 1
# do
# python run_example/pickplace/test_dynamics_pickplace.py \
#     --task pickplace_easy \
#     --num_workers 2 \
#     --behavior_epoch 30 --num_diffusion_iters $iter --horizon 40 \
#     --load_diffusion_path ${diff_path} --sample_ratio ${ratio} --task_weight 1 \
#     --load-dynamics-path ${dyn_path} \
#     --rollout_epochs 100 &
# done 
# wait

for ratio in 0.2 0.4 0.6 1
do
python run_example/pickplace/test_dynamics_pickplace.py \
    --task pickplace \
    --num_workers 1 \
    --behavior_epoch 30 --num_diffusion_iters $iter --horizon 40 \
    --load_diffusion_path ${diff_path} --sample_ratio ${ratio} --task_weight 0.8 \
    --load-dynamics-path ${dyn_path} \
    --rollout_epochs 100 &
done 
wait

# for weight in 1.2 1.4 1.6 1.8 2
# do
# python run_example/pickplace/test_dynamics_pickplace.py \
#     --task pickplace_easy \
#     --num_workers 1 \
#     --behavior_epoch 30 --num_diffusion_iters $iter --horizon 40 \
#     --load_diffusion_path ${diff_path} --sample_ratio 1 --task_weight 1 \
#     --load-dynamics-path ${dyn_path} \
#     --rollout_epochs 100 &
# done 
# wait

# for weight in 1.2 1.4 1.6 1.8 2
# do
# python run_example/pickplace/test_dynamics_pickplace.py \
#     --task pickplace \
#     --num_workers 1 \
#     --behavior_epoch 30 --num_diffusion_iters $iter --horizon 40 \
#     --load_diffusion_path ${diff_path} --sample_ratio ${ratio} --task_weight 1 \
#     --load-dynamics-path ${dyn_path} \
#     --rollout_epochs 100 &
# done 
# wait

# python run_example/pickplace/test_dynamics_pickplace.py \
#     --task pickplace_easy \
#     --num_workers 4 \
#     --behavior_epoch 30 --num_diffusion_iters $iter --horizon 40 \
#     --load_diffusion_path ${diff_path} --sample_ratio 0.8 --task_weight 2 \
#     --load-dynamics-path ${dyn_path} \
#     --rollout_epochs 100 
# python run_example/pickplace/test_dynamics_pickplace.py \
#     --task pickplace_easy \
#     --num_workers 4 \
#     --behavior_epoch 30 --num_diffusion_iters $iter --horizon 40 \
#     --load_diffusion_path ${diff_path} --sample_ratio 0.8 --task_weight 1 \
#     --load-dynamics-path ${dyn_path} \
#     --rollout_epochs 100 

# done

# python run_example/pickplace/run_mbrcsl_pickplace.py \
#     --behavior_epoch 30 --num_diffusion_iters 10 --horizon 40 \
#     --load_diffusion_path ${diff_path} \
#     --load-dynamics-path ${dyn_path} \
#     --rollout_ckpt_path ${rollout_path}

# python run_example/pickplace/run_onlinercsl_pickplace.py \
#     --behavior_epoch 30 --num_diffusion_iters 100 --horizon 40 \
#     --load_diffusion_path ${diff_path} \
#     --load-dynamics-path ${dyn_path} \
#     --rollout_ckpt_path ${rollout_path} --rollout-batch 1 --rollout_epochs 1000 --num_need_traj 40