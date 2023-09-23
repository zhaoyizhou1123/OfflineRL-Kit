# diff_path="logs/pickplace/diffusion/timestamp_23-0915-004443&0_keep/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0916-014735&0/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0916-030832&0/checkpoint/policy.pth"
# diff_path="logs/pickplace/diffusion/timestamp_23-0916-110230&0_w1.5/model/policy.pth"
# diff_path="logs/pickplace_easy/onlinercsl/rcsl_diff/timestamp_23-0917-052910&0/model/policy_best.pth"
# diff_path="logs/pickplace_easy/onlinercsl/rcsl_diff/timestamp_23-0917-073935&0/checkpoint/policy_best.pth"
# diff_path="logs/pickplace_easy/test_dyn_regress&sample_ratio=1.0&task_weight=1.0/diffusion/timestamp_23-0920-030643&0_keep/model/policy.pth"
# diff_path=None
# dyn_path="logs/pickplace/test_dyn/dynamics/timestamp_23-0915-015639-s0_keep/model"
# dyn_path="logs/pickplace_easy/mbrcsl_regress/dynamics/timestamp_23-0918-074944&0_keep/model"
# dyn_path="logs/pickplace_easy/test_dyn_regress&sample_ratio=1.0&task_weight=1.0/dynamics_regress/timestamp_23-0920-030636&0_keep/model"
# dyn_path="logs/pickplace/test_dyn_regress&sample_ratio=1.0&task_weight=1.0/dynamics_regress/timestamp_23-0920-040453&0_rev/model"
# dyn_path="logs/pickplace_easy/test_dyn_regress&sample_ratio=1.0&task_weight=1.0/dynamics_regress/timestamp_23-0920-090518&0/model"
# dyn_path="logs/pickplace_easy/test_dyn_regress&sample_ratio=1.0&task_weight=1.0/dynamics_regress/timestamp_23-0920-090518&0_sar_ok/model"
# dyn_path="logs/pickplace/test_dyn_trans&sample_ratio=0.8&task_weight=1.0/dynamics_trans/timestamp_23-0921-081528&0_keep/model"
# dyn_path="logs/pickplace/test_dyn_trans&n_layer=4&sample_ratio=0.8/dynamics_trans/timestamp_23-0921-093512&0_keepobs/model"
dyn_path=None
# rollout_path="logs/pickplace/test_dyn/rollout/timestamp_23-0915-061633&0_1000/checkpoint"
# rollout_path="logs/pickplace/test_dyn/rollout/timestamp_23-0915-054333&0_10000_sparse/checkpoint"
# rollout_path="logs/pickplace/test_dyn/rollout_true/timestamp_23-0915-211431&0_keep/checkpoint"
rollout_path="logs/pickplace_easy/test_dyn/rollout_true/timestamp_23-0916-222425&0_data/checkpoint"
# rollout_path=None

# for sample in 0.8 1
# do
#     for weight in 1.2 1
#     do
#     python run_example/pickplace/test_dynamics_regress_pickplace.py \
#         --task 'pickplace_easy' --horizon 40 \
#         --load-dynamics-path ${dyn_path} \
#         --load_diffusion_path ${diff_path} --num_diffusion_iters 5 --task_weight ${weight} --sample_ratio ${sample} \
#         --rollout_epochs 250 &
#     python run_example/pickplace/test_dynamics_regress_pickplace.py \
#         --task 'pickplace' --horizon 40 \
#         --load-dynamics-path ${dyn_path} \
#         --load_diffusion_path ${diff_path} --num_diffusion_iters 5 --task_weight ${weight} --sample_ratio ${sample} \
#         --rollout_epochs 250 &
#     done
#     wait
# done

# diff_path="logs/pickplace_easy/test_dyn_regress&sample_ratio=1.0&task_weight=1.0/diffusion/timestamp_23-0920-030643&0_keep/model/policy.pth"
# diff_path="logs/pickplace_easy/test_dyn&sample_ratio=0.8&task_weight=1.4/diffusion/timestamp_23-0920-062514&0_keep/model/policy.pth"
# task="pickplace"
# python run_example/pickplace/test_dynamics_trans_pickplace.py \
#     --seed 3 \
#     --task ${task} --algo-name "test_dyn_trans" --horizon 40 \
#     --load-dynamics-path ${dyn_path} --n_layer 4 \
#     --load_diffusion_path ${diff_path} --num_diffusion_iters 5 --task_weight 1 --sample_ratio 0.8 \
#     --rollout_epochs 100

diff_path="logs/pickplace_easy/test_dyn&sample_ratio=0.8&task_weight=1.4/diffusion/timestamp_23-0920-062514&0_keep/model/policy.pth"
task="pickplace"
python run_example/pickplace/test_dynamics_transv2_pickplace.py \
    --seed 3 \
    --task ${task} --algo-name "test_dyn_trans" --horizon 40 \
    --load-dynamics-path ${dyn_path} --n_layer 4 \
    --load_diffusion_path ${diff_path} --num_diffusion_iters 5 --task_weight 1 --sample_ratio 0.8 \
    --rollout_epochs 100



