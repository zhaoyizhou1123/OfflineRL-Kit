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

# for size in 8 16 32 64 128 256 512
# do
#     for arch in 8 16 32 64 128 256 512 
#     do
#         if [ "${arch}" -le "${size}" ]; then 
#             python run_example/linearq/run_rcsl_linearq.py --hidden-dims ${size} --rcsl-hidden-dims ${arch}
#             # python run_example/linearq/run_cql_linearq.py --hidden-dims ${size} --env_param ${arch}
#         fi
#     done
# done

# for size in 8 16 32 64 128 256 512
# do
#     for arch in 8 16 32 64 128 256 512 
#     do
#         if [ "${arch}" -le "${size}" ]; then 
#             # python run_example/linearq/run_rcsl_linearq.py --hidden-dims ${size} --rcsl-hidden-dims ${arch}
#             python run_example/linearq/run_cql_linearq.py --hidden-dims ${size} --env_param ${arch}
#         fi
#     done
# done

python run_example/linearq/run_cql_linearq.py --hidden-dims 16 --env_param 16 &
# python run_example/linearq/run_cql_linearq.py --hidden-dims 48 --env_param 16 &
# python run_example/linearq/run_cql_linearq.py --hidden-dims 32 --env_param 32 &
python run_example/linearq/run_cql_linearq.py --hidden-dims 16 --env_param 32 &
wait

python run_example/linearq/run_cql_linearq.py --hidden-dims 16 --env_param 64 &
# python run_example/linearq/run_cql_linearq.py --hidden-dims 192 --env_param 64 &
python run_example/linearq/run_cql_linearq.py --hidden-dims 16 --env_param 128 &
# python run_example/linearq/run_cql_linearq.py --hidden-dims 384 --env_param 128 &
wait
python run_example/linearq/run_cql_linearq.py --hidden-dims 16 --env_param 256
# python run_example/linearq/run_cql_linearq.py --hidden-dims 768 --env_param 256
wait