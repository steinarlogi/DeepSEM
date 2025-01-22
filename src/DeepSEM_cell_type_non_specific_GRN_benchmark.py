import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from src.Model import VAE_EAD
from src.utils import evaluate, extractEdgesFromMatrix

Tensor = torch.cuda.FloatTensor


class non_celltype_GRN_model:
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

        data = np.array(data, dtype=float)
        data = data.T
        data = (data - data.mean(0)) / (data.std(0))
        num_genes = len(gene_labels)
        dataloader = DataLoader(data, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)

        return dataloader, gene_labels, num_genes

    def train_model(self):
        opt = self.opt
        dataloader, gene_name, num_genes = self.init_data()

        adj_A_init = self.initalize_A(num_genes)
        vae = VAE_EAD(adj_A_init, 1, opt.n_hidden, opt.K).float().cuda()
        optimizer = optim.RMSprop(vae.parameters(), lr=opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)

        vae.train()
        for epoch in range(opt.n_epochs + 1):
            loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse = [], [], [], [], [], []
            if epoch % (opt.K1 + opt.K2) < opt.K1:
                vae.adj_A.requires_grad = False
            else:
                vae.adj_A.requires_grad = True
            for i, data_batch in enumerate(dataloader, 0):
                optimizer.zero_grad()
                inputs = data_batch
                inputs = Variable(inputs.type(Tensor))
                temperature = max(0.95 ** epoch, 0.5)
                loss, loss_rec, loss_gauss, loss_cat, dec, y, hidden = vae(inputs, dropout_mask=None,
                                                                           temperature=temperature, opt=opt)
                sparse_loss = opt.alpha * torch.mean(torch.abs(vae.adj_A))
                loss = loss + sparse_loss
                loss.backward()
                mse_rec.append(loss_rec.item())
                loss_all.append(loss.item())
                loss_kl.append(loss_gauss.item() + loss_cat.item())
                loss_sparse.append(sparse_loss.item())
                if epoch % (opt.K1 + opt.K2) < opt.K1:
                    optimizer.step()
                else:
                    optimizer2.step()
            scheduler.step()
            if epoch % (opt.K1 + opt.K2) >= opt.K1:
                print('epoch:', epoch, 'loss:',
                      np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                      np.mean(loss_sparse))
        extractEdgesFromMatrix(vae.adj_A.cpu().detach().numpy(), gene_name, None).to_csv(
            opt.save_name + '/GRN_inference_result.tsv', sep='\t', index=False)
