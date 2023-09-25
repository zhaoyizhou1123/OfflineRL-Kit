# dynamics_path="logs/pickplace_easy/mbrcsl_regress/dynamics/timestamp_23-0918-070448&0_keep/model"
dyn_path="logs/pickplace/combo/timestamp_23-0922-235459&0/model"
# dyn_path=None
# diff_path="logs/pickplace/diffusion/timestamp_23-0914-110329&0/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0915-004443&0_keep/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0916-110230&0_w1.5/model/policy.pth"
diff_path=None
# rollout_path="logs/pickplace_easy/test_dyn/rollout_true/timestamp_23-0916-222425&0_data/checkpoint"
rollout_path=None

for seed in 0 1 2 3
do
python run_example/pickplace/run_combo_pickplace.py \
    --task pickplace --seed ${seed} \
    --cql-weight 1 \
    --load-dynamics-path ${dyn_path} \
    --eval_episodes 100 --epoch 100 &
sleep 60
done
wait 
# for seed in 0 1 2 3
# do
# python run_example/pickplace/run_cql_pickplace.py \
#     --seed ${seed} \
#     --task pickplace \
#     --cql-weight 1 \
#     --eval_episodes 100 --epoch 200 &
# done
# wait

# python run_example/pickplace/run_regress_pickplace.py \
#     --num_workers 4 \
#     --algo-name rcsl_regress_gauss \
#     --task pickplace
