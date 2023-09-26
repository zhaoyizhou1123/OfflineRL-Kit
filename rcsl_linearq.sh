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

#  

for size in 16 32 48 64 80 96 112 128 144 160
do
    for arch in 16
    do
        if [ "${arch}" -le "${size}" ]; then 
            python run_example/linearq/run_rcsl_linearq.py --env_param ${size} --rcsl-hidden-dims ${arch} --num_workers 4 --rcsl-epoch 400 --algo-name "rcsl_revlinearq" 
            # python run_example/linearq/run_cql_linearq.py --env_param ${size} --hidden-dims ${arch}
        fi
    done
done

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