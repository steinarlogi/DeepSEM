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

parser = argparse.ArgumentParser('A script to tune the hyperparameters for the benchmarking data')

parser.add_argument('--data', help='The file containing the training data', required=True)

args = parser.parse_args()

Tensor = torch.cuda.FloatTensor


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

        print (config)
        data = np.array(data, dtype=float)
        data = data.T
        data = (data - data.mean(0)) / (data.std(0))
        num_genes = len(gene_labels)
        dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=1)

        return dataloader, gene_labels, num_genes


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
                
     
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')
            torch.save(
                (vae.state_dict(), optimizer.state_dict(), optimizer2.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report({
                'loss': loss.item()
            }, checkpoint=checkpoint)


def main():
    config = dict({
        'batch_size': 32,
        'n_hidden': 128,
        'K': 1,
        'lr': tune.loguniform(1e-5, 1e-2),
        'lr_step_size': 0.99,
        'gamma': tune.choice([0.90, .91, .92, .93, .94, .95, .96, .97, .98, .99]),
        'n_epochs': 90,
        'K1': 1,
        'K2': 2,
        'alpha': tune.sample_from(lambda _: int(np.random.random()* 20 + 90)),
        'beta': 1
        })

    scheduler = ASHAScheduler(
        max_t = 90,
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(infer_grn),
            resources={'cpu': 1, 'gpu': 1}
        ),
        tune_config=tune.TuneConfig(
            metric='loss',
            mode='min',
            scheduler=scheduler,
            num_samples=30
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result('loss', 'min')

    print (f'Best trial config {best_result.config}')
    print (f'Best trial final training loss {best_result.metrics["loss"]}')




if __name__ == '__main__':
    print ('Running the parameter search...')
    main()