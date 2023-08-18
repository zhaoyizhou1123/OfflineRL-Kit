# d_path="log/maze/combo/seed_1&timestamp_23-0809-143727/model"
# algo=mopo
cql_weight=1 # combo
penalty_coef=1 # mopo
epoch=50

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

python run_example/run_cql.py