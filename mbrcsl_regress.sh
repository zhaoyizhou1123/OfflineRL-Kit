# dynamics_path="logs/pickplace_easy/mbrcsl_regress/dynamics/timestamp_23-0918-070448&0_keep/model"
# dyn_path="logs/pickplace/test_dyn/dynamics/timestamp_23-0915-015639-s0_keep/model"
dyn_path=None
# diff_path="logs/pickplace/diffusion/timestamp_23-0914-110329&0/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0915-004443&0_keep/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0916-110230&0_w1.5/model/policy.pth"
diff_path=None
# rollout_path="logs/pickplace_easy/test_dyn/rollout_true/timestamp_23-0916-222425&0_data/checkpoint"
rollout_path=None

# for task_weight in 1.5 1.75 1.25 1.9 1.1
# do
#     python run_example/pickplace/run_diffusion_pickplace.py \
#         --eval_episodes 10 --load_diffusion_path ${diff_path} --num_diffusion_iters 5 --horizon 40 --rcsl-epoch 100 \
#         --task_weight ${task_weight}
# done

# for task_weight in 1.5
# do
#     python run_example/pickplace/run_regress_pickplace.py \
#         --task 'pickplace_easy' \
#         --rollout_ckpt_path ${rollout_path} \
#         --eval_episodes 50 --horizon 40 --rcsl-epoch 100 \
#         --task_weight ${task_weight}
# done

for sample in 0.3 0.4 0.5
do
python run_example/pickplace/run_mbrcsl_pickplace_v2.py \
    --num_workers 4 \
    --task 'pickplace_easy' --horizon 40 \
    --load-dynamics-path ${dyn_path} \
    --load_diffusion_path ${diff_path} --behavior_epoch 30 --sample_ratio ${sample} \
    --rollout_ckpt_path ${rollout_path} --rollout_epochs 30000 --num_need_traj 2500 --rollout-batch 1 \
    --eval_episodes 50  --rcsl-epoch 75
done

# for sample in 0.3
# do
# python run_example/pickplace/run_mbrcsl_pickplace_v2.py \
#     --num_workers 4 \
#     --task 'pickplace_easy' --horizon 40 \
#     --load-dynamics-path ${dyn_path} \
#     --load_diffusion_path ${diff_path} --behavior_epoch 1 --sample_ratio ${sample} \
#     --rollout_ckpt_path ${rollout_path} --rollout_epochs 30000 --num_need_traj 2500 --rollout-batch 1 \
#     --eval_episodes 50  --rcsl-epoch 75
# done

# d_path="log/maze/combo/seed_1&timestamp_23-0809-143727/model"
# algo=mopo
# cql_weight=1 # combo
# penalty_coef=1 # mopo
# epoch=50

# for seed in 0 1 2 3
# do
# python run_example/run_combo.py --seed ${seed} \
#     --cql-weight ${cql_weight} \
#     --epoch ${epoch}

# python run_example/run_mopo.py --seed ${seed} \
#     --penalty-coef ${penalty_coef}\
#     --epoch ${epoch}
#     # --load-dynamics-path ${d_path} \

# done

# for size in 128 
# do
#     for arch in 12 14 16
#     do
#         if [ "${arch}" -le "${size}" ]; then 
#             python run_example/linearq/run_rcsl_linearq.py --env_param ${size} --rcsl-hidden-dims ${arch}
#             # python run_example/linearq/run_cql_linearq.py --env_param ${size} --hidden-dims ${arch}
#         fi
#     done
# done

# for size in 8 16 32 64 128 256 512
# do
#     for arch in 8 16 32 64 128 256 512 
#     do
#         if [ "${arch}" -le "${size}" ]; then 
#             # python run_example/linearq/run_rcsl_linearq.py --env_param ${size} --rcsl-hidden-dims ${arch}
#             python run_example/linearq/run_cql_linearq.py --env_param ${size} --hidden-dims ${arch}
#         fi
#     done
# done