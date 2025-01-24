#!/bin/bash

if [ ! -d "$1" ]; then
    echo "$1 is not a directory"
    exit 1
fi

for file in ./$1/*; do
    if [ -f "$file" ] && [[ "$file" == *GeneExpression.csv ]]; then
        filename=$(basename $file)
        network_out_name="${filename%GeneExpression.csv}grn.tsv"
        python3 main.py --task non_celltype_GRN_benchmark --data_file "$file" --setting best_params --alpha 100 --beta 1 --n_epoch 90 --save_name grn_benchmark_out
        if [ ! -f "./grn_benchmark_out/$network_out_name" ]; then
            echo ".grn_benchmark_out/$network_out_name"
            echo "Network out name does not exists"
            exit 1
        fi
        python3 deep_sem_grn_to_grn_benchmark.py --from_file "./grn_benchmark_out/$network_out_name" --save_dir ./grn_benchmark_out
        rm "./grn_benchmark_out/$network_out_name"
    fi
done

