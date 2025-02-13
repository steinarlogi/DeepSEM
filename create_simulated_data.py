import argparse
import numpy as np
import pandas as pd
import genesnake as gs
import os


parser = argparse.ArgumentParser()
parser.add_argument('--outputdir', required=True, help='output directory')

args = parser.parse_args()

output_dir = args.outputdir

try:
    os.mkdir(output_dir)
except:
    print ('Output already exists')

### Simulation
genes = 100 # number of genes
spars = 4 # sparsity

network = gs.grn.make_DAG(genes, spars, self_loops=True)
M = gs.GRNmodel.make_model(network)
M.set_pert('diag', effect=(0.9, 1), noise=0.01, reps=3, sign=-0.1)
M.simulate_data(exp_type='ss', SNR=10, noise_model='microarray')

gene_labels = [f'G{number:03}' for number in range(1, genes + 1)]

### GRN inference
Y = M.data # simulated expression data
P = M.perturbation # perturbation (P) matrix
Y.index = gene_labels
P.index = gene_labels
true_network = M.network
true_network.index = gene_labels
true_network.columns = gene_labels


Y.to_csv(os.path.join(output_dir, 'data.csv'))
P.to_csv(os.path.join(output_dir, 'perturbations.csv'))
true_network.to_csv(os.path.join(output_dir, 'true_grn.csv'))
