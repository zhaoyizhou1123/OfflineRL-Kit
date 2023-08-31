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

algo=mbrcsl
horizon=1000
d_path="logs/halfcheetah-medium-v2/mbrcsl/dynamics/seed_0&timestamp_23-0824-020002_keep/model"
# d_path=None
diff_seed="test_rollout"
diff_path="logs/halfcheetah-medium-v2/mbrcsl/diffusion/test_rollout"
rollout_ckpt_path="logs/halfcheetah-medium-v2/rollout/test"
rollout_batch=20

python run_example/run_mbrcsl.py \
    --horizon ${horizon} \
    --load-dynamics-path ${d_path} \
    --diffusion_seed ${diff_seed} \
    --load_diffusion_path ${diff_path} \
    --rollout_ckpt_path ${rollout_ckpt_path} \
    --rollout-batch ${rollout_batch}
