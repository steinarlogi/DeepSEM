# Information

This document contains information about how to run the scripts.

## deep_sem_to_grn_benchmark.py
This is a script used to convert the output from the DeepSEM algorithm to a format that is understood by grnbenchmark.org

(Use --help for information about the arguments)

## create_simulated_data.py
A script to create synthetic expression data with perturbations using geneSPIDER.

(Use --help for information about the arguments)

## hyperparameter_search.py
A script that uses ray to search for parameters. I will change it so that it uses the auroc and aupr as measures to optimize instead of the loss.

(Use --help for information about the arguments)

## hyperparameter_search_p.py
Similar to `hyperparameter_search_p.py` but for the p-based loss model.

(Use --help for information about the arguments)

## benchmark.py
A script that uses genesnake to benchmark a given gene regulatory network against a true network

(Use --help for information about the arguments)

## grn_benchmark.sh
**Usage**: `./grn_benchmark.sh dir_name`

A script that loops through the given directory of GRNBenchmark data and infers the GRNs for each of them using the normal DeepSEM algorithm. The output is saved to a folder called grn_benchmark_out. The script automatically converts the deepsem output to a format that is understood by grnbenchmark.org using the script deep_sem_to_grn_benchmark.py

## grn_benchmark_perturb.sh
**Usage:** `./grn_benchmark.sh dir_name`

Similiar to the grn_benchmark.sh script. However it uses the new DeepSEM algorithm with perturb loss. The output is saved to a folder called grn_benchmark_out_perturb.