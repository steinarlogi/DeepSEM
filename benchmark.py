import argparse
import numpy as np
import pandas as pd
import genesnake as gs

parser = argparse.ArgumentParser()
parser.add_argument('--inferred_grn', required=True, help='The file that contains the inferred')
parser.add_argument('--true_grn', required=True, help='The file that contains the true GRN')

args = parser.parse_args()

true_grn_file = args.true_grn
grn = args.inferred_grn

n_genes = 50
gene_names = [f'G{number:03}' for number in range(1, n_genes + 1)]
gene_name_to_idx = { f'G{number:03}': number - 1 for number in range(1, n_genes + 1) }
adj_matrix = np.zeros((n_genes, n_genes))

true_grn = pd.read_csv(true_grn_file, index_col=0)

# Create a pandas dataframe adjacency matrix from the edgelist
with open(grn, mode='r') as f:
    lines = f.readlines()

    for i in range(1, len(lines)):
        line = lines[i].split(',')
        regulator, target, weight, sign = line
        regulator_idx = gene_name_to_idx[regulator]
        target_idx = gene_name_to_idx[target]
        weight, sign = float(weight), float(sign)
        adj_matrix[regulator_idx, target_idx] = weight

A = pd.DataFrame(adj_matrix)
A.columns = gene_names
A.index = gene_names

stats = gs.benchmarking.benchmark(
	estimated_network = A,
	true_network = true_grn,
	output_dir = 'simulated_data',
	model_name = 'DeepSEM',
	)
