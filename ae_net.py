import gc
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import helper_hpc as helper
import time

# DEFINE a CONV NN

class AE(pl.LightningModule):
    def __init__(self, encoded_space_dim, diversity=None, lr=.001):
        super().__init__()
        self.save_hyperparameters()
        # Encoder
        self.BatchNorm1 = nn.BatchNorm2d(32)
        self.BatchNorm2 = nn.BatchNorm2d(128)
        self.BatchNorm3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, encoded_space_dim)
        self.dropout1 = nn.Dropout2d(0.05)
        self.dropout2 = nn.Dropout2d(0.1)
        self.conv_layers = nn.ModuleList([nn.Conv2d(3, 32, 3, padding=1), 
                            nn.Conv2d(32, 64, 3, padding=1), 
                            nn.Conv2d(64, 128, 3, padding=1), 
                            nn.Conv2d(128, 128, 3, padding=1), 
                            nn.Conv2d(128, 256, 3, padding=1), 
                            nn.Conv2d(256, 256, 3, padding=1)])
        self.activations = {}
        for i in range(len(self.conv_layers)):
            self.activations[i] = []

        # Decoder
        self.t_conv_layers = nn.ModuleList([nn.ConvTranspose2d(256, 256, 3, padding=1),
                                            nn.ConvTranspose2d(256, 128, 3, padding=1),
                                            nn.ConvTranspose2d(128, 128, 3, padding=1),
                                            nn.ConvTranspose2d(128, 64, 3, padding=1),
                                            nn.ConvTranspose2d(64, 32, 3, padding=1),
                                            nn.ConvTranspose2d(32, 3, 3, padding=1)])
        self.t_BatchNorm3 = nn.BatchNorm2d(256)
        self.t_BatchNorm2 = nn.BatchNorm2d(128)
        self.t_BatchNorm1 = nn.BatchNorm2d(32)

        self.t_upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.t_unflatten = nn.Unflatten(1, (256, 4, 4))

        self.t_dropout2 = nn.Dropout2d(0.1)
        self.t_dropout1 = nn.Dropout2d(0.05)

        self.t_fc3 = nn.Linear(encoded_space_dim, 512)
        self.t_fc2 = nn.Linear(512, 1024)
        self.t_fc1 = nn.Linear(1024, 4096)

        self.diversity = diversity
        self.loss_fn = torch.nn.MSELoss()
        # self.avg_novelty = 0

    def forward(self, x, get_activations=False):
        # Encode
        conv_count = 0
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        conv_count += 1
        x = self.BatchNorm1(x)
        x = F.relu(x)
        
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        conv_count += 1
        x = F.relu(x)
        
        x = self.pool(x)
        
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        conv_count += 1
        x = self.BatchNorm2(x)
        x = F.relu(x)
        
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        conv_count += 1
        x = F.relu(x)
        
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        conv_count += 1
        x = self.BatchNorm3(x)
        x = F.relu(x)
        
        x = self.conv_layers[conv_count](x)
        if get_activations:
            self.activations[conv_count].append(x)
        x = F.relu(x)
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout2(x)
        
        x = self.fc1(x) # 4096 -> 1024
        x = F.relu(x)
        
        x = self.fc2(x) # 1024 -> 512
        x = F.relu(x)
        
        x = self.dropout2(x)
        
        x = self.fc3(x) # 512 -> encoding size

        # Decode
        t_conv_count = 0
        x = self.t_fc3(x) # encoding -> 512
        x = F.relu(x)

        x = self.t_dropout1(x)
        
        x = self.t_fc2(x) # 512 -> 1024
        x = F.relu(x)
        
        x = self.t_fc1(x) # 1024 -> 4096
        x = F.relu(x)
        
        x = self.t_dropout2(x)
        x = self.t_unflatten(x)
        x = self.t_upsample(x)
        
        x = self.t_conv_layers[t_conv_count](x)
        t_conv_count += 1
        x = self.t_BatchNorm3(x)
        x = F.relu(x)

        x = self.t_conv_layers[t_conv_count](x)
        t_conv_count += 1
        x = F.relu(x)
        x = self.t_dropout1(x)
        x = self.t_upsample(x)

        x = self.t_conv_layers[t_conv_count](x)
        t_conv_count += 1
        x = self.t_BatchNorm2(x)
        x = F.relu(x)
        
        x = self.t_conv_layers[t_conv_count](x)
        t_conv_count += 1
        x = F.relu(x)
        x = self.t_upsample(x)

        x = self.t_conv_layers[t_conv_count](x)
        t_conv_count += 1
        x = self.t_BatchNorm1(x)
        x = F.relu(x)

        x = self.t_conv_layers[t_conv_count](x)


        return x

        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        # get loss
        loss = self.loss_fn(logits, x)
        # get acc
        # labels_hat = torch.argmax(logits, 1)
        # acc = torch.sum(y==labels_hat)/(len(x)*1.0)
        # log loss and acc
        self.log('train_loss', loss)
        # self.log('train_acc', acc)
        batch_dictionary={
	            "train_loss": loss, 'loss': loss
	        }
        return batch_dictionary

    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        
        self.log('train_loss_epoch', avg_loss)
        # self.log('train_acc_epoch', avg_acc)
        gc.collect()
    
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            if batch_idx == 0:
                self.step=1
            else:
                self.step=2
            x, y = val_batch
            logits = self.forward(x, get_activations=True)
            # get loss
            loss = self.loss_fn(logits, x)

            novelty_score = self.compute_feature_novelty()
            # clear out activations
            for i in range(len(self.conv_layers)):
                self.activations[i] = []
            # log loss, acc, class acc, and novelty score
            self.log('val_loss', loss)
            self.log('val_novelty', novelty_score)
            batch_dictionary = {'val_loss': loss, 'val_novelty': novelty_score}
        return batch_dictionary
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_novelty = np.stack([x['val_novelty'] for x in outputs]).mean()
        self.avg_novelty = avg_novelty
        self.log('val_loss_epoch', avg_loss)
        self.log('val_novelty_epoch', avg_novelty)
        gc.collect()

    def test_step(self, test_batch, batch_idx):
        with torch.no_grad():
            x, y = test_batch
            logits = self.forward(x, get_activations=True)
            # get loss
            loss = self.loss_fn(logits, x)
            # get novelty score
            novelty_score = self.compute_feature_novelty()
            # clear out activations
            for i in range(len(self.conv_layers)):
                self.activations[i] = []
            # log loss, acc, class acc, and novelty score
            self.log('test_loss', loss)
            self.log('test_novelty', novelty_score)
            batch_dictionary = {'test_loss': loss, 'test_novelty': novelty_score}
        return batch_dictionary

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_novelty = np.stack([x['test_novelty'] for x in outputs]).mean()
        self.avg_novelty = avg_novelty
        self.log('test_loss_epoch', avg_loss)
        self.log('test_novelty_epoch', avg_novelty)
        gc.collect()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        # return F.nll_loss(logits, labels)
        return F.cross_entropy(logits, labels)
        
    def set_filters(self, filters):
        for i in range(len(filters)):
            self.conv_layers[i].weight.data = filters[i]
    
    def get_filters(self, numpy=False):
        if numpy:
            return [m.weight.data.detach().cpu().numpy() for m in self.conv_layers]
        return [m.weight.data.detach().cpu() for m in self.conv_layers]

    def compute_feature_novelty(self):
        
        # start = time.time()
        # layer_totals = {}
        # with torch.no_grad():
        #     # for each conv layer 4d (batch, channel, h, w)
        #     for layer in range(len(self.activations)):
        #         B = len(self.activations[layer][0])
        #         C = len(self.activations[layer][0][0])
        #         a = self.activations[layer][0]
        #         layer_totals[layer] = torch.abs(a.unsqueeze(2) - a.unsqueeze(1)).sum().item()
        # end = time.time()
        # print('gpu answer: {}'.format(sum(layer_totals.values())))
        # print('gpu time: {}'.format(end-start))
        # return(sum(layer_totals.values()))

            # layer_totals[layer] = np.abs(np.expand_dims(a, axis=2) - np.expand_dims(a, axis=1)).sum().item()

        l = []
        for i in self.activations:
            self.activations[i][0] = self.activations[i][0].detach().cpu().numpy()
            if self.diversity=='relative':
                l.append(helper.diversity_relative(self.activations[i][0]))
            elif self.diversity=='original':
                l.append(helper.diversity_orig(self.activations[i]))
            elif self.diversity=='absolute':
                l.append(helper.diversity(self.activations[i][0]))
            elif self.diversity=='cosine':
                l.append(helper.diversity_cosine_distance(self.activations[i][0]))
            elif self.diversity == 'constant':
                l.append(helper.diversity_constant(self.activations[i][0]))
            else:
                l.append(helper.diversity(self.activations[i][0]))

        if self.step==1:
            print([max(np.abs(x.flatten())) for x in self.get_filters(numpy=True)])
        return(sum(l))

