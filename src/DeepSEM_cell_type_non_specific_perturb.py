import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from src.PModel import VAE_EAD
from src.utils import evaluate, extractEdgesFromMatrix

Tensor = torch.cuda.FloatTensor


class non_celltype_GRN_model_perturb:
    def __init__(self, opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')

    def initalize_A(self, num_genes):
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + (np.random.rand(num_genes * num_genes) * 0.0002).reshape(
            [num_genes, num_genes])
        for i in range(len(A)):
            A[i, i] = 0
        return A

    def initialize_A_with_perturb(self, P, Y):
        A = np.matmul(P.T, np.linalg.pinv(Y.T))
        for i in range(len(A)):
            A[i, i] = 0
        return A


    def init_data(self):
        # Read the expression data from csv file
        gene_labels = []
        data = []
        with open(self.opt.data_file) as f:
            lines = f.readlines()

            for i in range(1, len(lines)):
                line = lines[i].split(',')

                gene_labels.append(line[0])
                data.append(line[1:])

        perturb_data = []
        with open(self.opt.perturb_file) as f:
            lines = f.readlines()

            for i in range(1, len(lines)):
                line = lines[i].split(',')
                perturb_data.append(line[1:])

        data = np.array(data, dtype=float)
        data = data.T
        data = (data - data.mean(0)) / (data.std(0))
        perturb_data = np.array(perturb_data, dtype=float)
        perturb_data = perturb_data.T
        num_genes = len(gene_labels)
        dataset = TensorDataset(torch.tensor(data), torch.tensor(perturb_data))
        dataloader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)

        return dataloader, gene_labels, num_genes, data, perturb_data

    def train_model(self):
        opt = self.opt
        dataloader, gene_name, num_genes, data, perturb_data = self.init_data()

        adj_A_init = self.initialize_A_with_perturb(perturb_data, data)
        vae = VAE_EAD(adj_A_init, 1, opt.n_hidden, opt.K).float().cuda()
        optimizer = optim.RMSprop(vae.parameters(), lr=opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)

        vae.train()
        for epoch in range(opt.n_epochs + 1):
            loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse, losses_perturb = [], [], [], [], [], [], []
            if epoch % (opt.K1 + opt.K2) < opt.K1:
                vae.adj_A.requires_grad = False
            else:
                vae.adj_A.requires_grad = True
            for i, data_batch in enumerate(dataloader, 0):
                data_batch, data_perturb = data_batch
                optimizer.zero_grad()
                optimizer2.zero_grad()
                inputs = data_batch
                inputs = Variable(inputs.type(Tensor))
                data_perturb = Variable(data_perturb.type(Tensor))
                temperature = max(0.95 ** epoch, 0.5)
                loss, loss_rec, loss_gauss, loss_cat, loss_perturb, dec, y, hidden = vae(inputs, data_perturb, dropout_mask=None,
                                                                           temperature=temperature, opt=opt)
                sparse_loss = opt.alpha * torch.mean(torch.abs(vae.adj_A))
                loss = loss + sparse_loss
                #loss = loss_perturb * 0.001 # TODO: REMOVE THIS
                loss.backward()
                #for name, param in vae.named_parameters():
                #    if param.grad is not None:
                #        print(f"{name} gradient norm: {param.grad.norm()}")
                mse_rec.append(loss_rec.item())
                loss_all.append(loss.item())
                loss_kl.append(loss_gauss.item() + loss_cat.item())
                losses_perturb.append(loss_perturb.item())
                loss_sparse.append(sparse_loss.item())
                if epoch % (opt.K1 + opt.K2) < opt.K1:
                    optimizer.step()
                else:
                    optimizer2.step()
            scheduler.step()
            if epoch % (opt.K1 + opt.K2) >= opt.K1:
                print('epoch:', epoch, 'loss:',
                      np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                      np.mean(loss_sparse), 'perturb_loss:', np.mean(losses_perturb))


        # Set the diagonal of the adjacency matrix to zero
        adj_A = vae.adj_A.cpu().detach().numpy()
        for i in range(adj_A.shape[0]):
            adj_A[i, i] = 0
        extractEdgesFromMatrix(adj_A, gene_name, None).to_csv(
            opt.save_name + f'/{str(os.path.splitext(str(os.path.split(opt.data_file)[1]))[0]).removesuffix("_GeneExpression")}_grn.tsv', sep='\t', index=False)
