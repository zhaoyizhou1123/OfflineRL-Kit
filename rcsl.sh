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

algo=rcsl
horizon=1000
num_workers=1
epoch=1000
arch="200 200 200 200"
batch=1024

task="hopper-medium-v2"
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=0 python run_example/run_${algo}.py \
        --task ${task} \
        --algo-name ${algo} \
        --seed ${seed} \
        --horizon ${horizon} \
        --rcsl-epoch ${epoch} \
        --rcsl-hidden-dims ${arch} \
        --rcsl-batch ${batch} \
        --num_workers ${num_workers} &
    sleep 40
done


task="walker2d-medium-v2" 
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=1 python run_example/run_${algo}.py \
        --task ${task} \
        --algo-name ${algo} \
        --seed ${seed} \
        --horizon ${horizon} \
        --rcsl-epoch ${epoch} \
        --rcsl-hidden-dims ${arch} \
        --rcsl-batch ${batch} \
        --num_workers ${num_workers} &
    sleep 40
done

task="halfcheetah-medium-v2"
for seed in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=2 python run_example/run_${algo}.py \
        --task ${task} \
        --algo-name ${algo} \
        --seed ${seed} \
        --horizon ${horizon} \
        --rcsl-epoch ${epoch} \
        --rcsl-hidden-dims ${arch} \
        --rcsl-batch ${batch} \
        --num_workers ${num_workers} &
    sleep 40
done

wait
