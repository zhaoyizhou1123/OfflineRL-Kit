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
dataset=../D4RL/dataset/halfcheetah/output_simple.hdf5
# dataset=None
horizon=1000
d_path="logs/halfcheetah-medium-v2/mbrcsl/dynamics/timestamp_23-0910-102245&0_mod_keep/model"
# d_path=None
diff_seed="test_diff"
diff_path="logs/halfcheetah-medium-v2/mbrcsl/diffusion/seed_test_diff&timestamp_23-0911-091333_mod_simple"
diff_epoch=50
# diff_path=None
rollout_ckpt_path="logs/halfcheetah-medium-v2/rollout/test"
rollout_batch=20

python run_example/test_dynamics.py \
    --dataset ${dataset} \
    --horizon ${horizon} \
    --load-dynamics-path ${d_path} \
    --diffusion_seed ${diff_seed} \
    --load_diffusion_path ${diff_path} \
    --behavior_epoch ${diff_epoch} \
    --rollout_ckpt_path ${rollout_ckpt_path} \
    --rollout-batch ${rollout_batch}
