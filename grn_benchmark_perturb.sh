#!/bin/bash

if [ ! -d "$1" ]; then
    echo "$1 is not a directory"
    exit 1
fi

for file in ./$1/*; do
    if [ -f "$file" ] && [[ "$file" == *GeneExpression.csv ]]; then
        filename=$(basename $file)
        network_out_name="${filename%GeneExpression.csv}grn.tsv"
        python3 main.py --task perturb --data_file "$file" --perturb_file "${file%GeneExpression.csv}Perturbations.csv" --setting best_params --save_name grn_benchmark_out_perturb
        if [ ! -f "./grn_benchmark_out_perturb/$network_out_name" ]; then
            echo ".grn_benchmark_out_perturb/$network_out_name"
            echo "Network out name, $network_out_name , does not exists"
            exit 1
        fi  
        python3 deep_sem_grn_to_grn_benchmark.py --from_file "./grn_benchmark_out_perturb/$network_out_name" --save_dir ./grn_benchmark_out_perturb
        rm "./grn_benchmark_out_perturb/$network_out_name"
    fi
done

