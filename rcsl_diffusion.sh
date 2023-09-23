# diff_path="logs/pickplace/diffusion/timestamp_23-0914-110329&0/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0915-004443&0_keep/checkpoint/policy.pth"
diff_path=None
# rollout_path="logs/pickplace_easy/test_dyn/rollout_true/timestamp_23-0916-222425&0_data/checkpoint"

task_weight=1.4
for seed in 0 1
do
    CUDA_VISIBLE_DEVICES=0 python run_example/pickplace/run_diffusion_pickplace.py \
        --seed ${seed} \
        --task pickplace --num_workers 2 \
        --eval_episodes 100 --load_diffusion_path ${diff_path} --num_diffusion_iters 5 --horizon 40 --rcsl-epoch 200 \
        --task_weight ${task_weight} &
done

for seed in 2 3
do
    CUDA_VISIBLE_DEVICES=1 python run_example/pickplace/run_diffusion_pickplace.py \
        --seed ${seed} \
        --task pickplace --num_workers 2 \
        --eval_episodes 100 --load_diffusion_path ${diff_path} --num_diffusion_iters 5 --horizon 40 --rcsl-epoch 200 \
        --task_weight ${task_weight} &
done
wait

# for task_weight in 1.5
# do
#     python run_example/pickplace/run_regress_pickplace.py \
#         --task 'pickplace_easy' \
#         --rollout_ckpt_path ${rollout_path} \
#         --eval_episodes 50 --horizon 40 --rcsl-epoch 100 \
#         --task_weight ${task_weight}
# done

# python run_example/pickplace/run_regress_ctg_pickplace.py \
#     --task 'pickplace_easy' \
#     --eval_episodes 100 --horizon 40 --rcsl-epoch 1000 \
#     --output_bins 512

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