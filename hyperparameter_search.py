from torch.utils.data import DataLoader
import torch
from torch import optim
from torch.autograd import Variable
import ray
from ray import train, tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import argparse
import numpy as np
from src.Model import VAE_EAD
import os
import tempfile
from types import SimpleNamespace
import genesnake as gs
from copy import deepcopy
import pandas as pd
import tempfile


parser = argparse.ArgumentParser('A script to tune the hyperparameters for the benchmarking data')

parser.add_argument('--data', help='The file containing the training data', required=True)
parser.add_argument('--true_grn_data', help='The file containing the true grn', required=True)

args = parser.parse_args()

Tensor = torch.cuda.FloatTensor

def benchmark(estimated_network, true_network):
    cutoffs = gs.benchmarking.calculate_cutoffs(
	estimated_network,
	)
    # Retain a copy of the original estimated network, since
    # cut_network operates inplace to avoid repeatedly copying the network.
    network = deepcopy(estimated_network)
    ms = []
    for cutoff in cutoffs:
        gs.benchmarking.cut_network(network, cutoff)
        m = gs.benchmarking.compare_networks(
            [network], true_network,
            exclude_diag = False,
            include_sign = False)
        ms.append(m)
    metrics = pd.concat(ms, axis = 0, ignore_index = True)

    with tempfile.TemporaryDirectory() as f:
        areas = gs.benchmarking.auroc_and_aupr(
            full_network = estimated_network,
            true_network = true_network,
            metrics = metrics,
            model_name = 'deepsem',
            output_dir = f,
            fix_ylim = False)

    return {**areas}



def load_data(config):
        # Read the expression data from csv file
        gene_labels = [] 
        data = []
        with open(args.data) as f:
            lines = f.readlines()
            
            for i in range(1, len(lines)):
                line = lines[i].split(',')

                gene_labels.append(line[0])
                data.append(line[1:])

        data = np.array(data, dtype=float)
        data = data.T
        data = (data - data.mean(0)) / (data.std(0))
        num_genes = len(gene_labels)
        dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=1)

        return dataloader, gene_labels, num_genes

def load_true_grn():
    true_grn = pd.read_csv(args.true_grn_data, index_col=0)
    return true_grn


def initialize_A(num_genes):
    A = np.ones([num_genes, num_genes]) / (num_genes - 1) + (np.random.rand(num_genes * num_genes) * 0.0002).reshape(
            [num_genes, num_genes])
    for i in range(len(A)):
        A[i, i] = 0
    return A


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def infer_grn(config):
    config = dotdict(config)
    dataloader, gene_name, num_genes = load_data(config)
    true_grn = load_true_grn()

    adj_A_init = initialize_A(num_genes)
    vae = VAE_EAD(adj_A_init, 1, config.n_hidden, config.K).float().cuda()
    optimizer = optim.RMSprop(vae.parameters(), lr=config.lr)
    optimizer2 = optim.RMSprop([vae.adj_A], lr=config.lr * 0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.gamma)

    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state, optimizer2_state, adj_state = torch.load(
                os.path.join(loaded_checkpoint_dir, 'checkpoint.pt')
            )
            vae.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            optimizer2.load_state_dict(optimizer2_state)
            adj_A_init = adj_state

    vae.train()
    for epoch in range(config.n_epochs+ 1):
        loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse = [], [], [], [], [], []
        if epoch % (config.K1 + config.K2) < config.K1:
            vae.adj_A.requires_grad = False
        else:
            vae.adj_A.requires_grad = True
        for i, data_batch in enumerate(dataloader, 0):
            optimizer.zero_grad()
            inputs = data_batch
            inputs = Variable(inputs.type(Tensor))
            temperature = max(0.95 ** epoch, 0.5)
            loss, loss_rec, loss_gauss, loss_cat, dec, y, hidden = vae(inputs, dropout_mask=None,
                                                                        temperature=temperature, opt=config)
            sparse_loss = config.alpha * torch.mean(torch.abs(vae.adj_A))
            loss = loss + sparse_loss
            loss.backward()
            mse_rec.append(loss_rec.item())
            loss_all.append(loss.item())
            loss_kl.append(loss_gauss.item() + loss_cat.item())
            loss_sparse.append(sparse_loss.item())
            if epoch % (config.K1 + config.K2) < config.K1:
                optimizer.step()
            else:
                optimizer2.step()
        scheduler.step()
        if epoch % (config.K1 + config.K2) >= config.K1:
            pass
            print('epoch:', epoch, 'loss:',
                    np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                    np.mean(loss_sparse))
                
     
        # Calculate the auroc and aupr
        stats = benchmark(pd.DataFrame(vae.adj_A.detach().cpu().numpy()), true_grn)
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')
            torch.save(
                (vae.state_dict(), optimizer.state_dict(), optimizer2.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report({
                'aupr': stats['AUPR'],
                'loss': np.mean(loss_all)
            }, checkpoint=checkpoint)


def main():
    config = dict({
        'batch_size': 64,
        'n_hidden': 256,
        'K': 1,
        'lr': .0031554350481570285,
        'lr_step_size': 0.99,
        'gamma': 0.94,
        'n_epochs': 90,
        'K1': 1,
        'K2': 2,
        'alpha': 105,
        'beta': tune.sample_from(lambda _: np.random.random() * 10 + 0.1)
        #'batch_size': tune.choice([32, 64, 128]),
        #'n_hidden': tune.choice([64, 128, 256, 512]),
        #'K': 1,
        #'lr': tune.loguniform(1e-5, 1e-2),
        #'lr_step_size': 0.99,
        #'gamma': tune.choice([0.90, .91, .92, .93, .94, .95, .96, .97, .98, .99]),
        #'n_epochs': 90,
        #'K1': 1,
        #'K2': 2,
        #'alpha': tune.sample_from(lambda _: int(np.random.random()* 20 + 90)),
        #'beta': 1
        })

    scheduler = ASHAScheduler(
        max_t = 90,
        grace_period=5,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(infer_grn, {'gpu': 1, 'cpu': 2}),
        tune_config=tune.TuneConfig(
            metric='aupr',
            mode='max',
            scheduler=scheduler,
            num_samples=100
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result('aupr', 'max')

    print (f'Best trial config {best_result.config}')
    print (f'Best trial final training loss {best_result.metrics["loss"]}')
    print (f'Best trial final aupr {best_result.metrics["aupr"]}')




if __name__ == '__main__':
    print ('Running the parameter search...')
    main()