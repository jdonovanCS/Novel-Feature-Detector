import gc
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import helper_hpc as helper
import torchmetrics

# DEFINE a CONV NN

class Net(pl.LightningModule):
    def __init__(self, num_classes=10, classnames=None, diversity=None, lr=.001, size=6):
        super().__init__()
        self.save_hyperparameters()
        
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

        if size % 2 == 1:
            size+=1

        self.conv_layers = nn.ModuleList([nn.Conv2d(3, 64, 3, padding=1), nn.Conv2d(64,64,3,padding=1)])
        if size > 2:
            self.conv_layers.extend([nn.Conv2d(64,128,3,padding=1), nn.Conv2d(128,128,3,padding=1)])
        if size > 4:
            self.conv_layers.extend([nn.Conv2d(128,256,3,padding=1), nn.Conv2d(256,256,3,padding=1)])
        
        self.activations = {}
        for i in range(size):
            self.activations[i] = []

        self.classnames = classnames
        self.diversity = diversity
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = lr
        # self.avg_novelty = 0

    def forward(self, x, get_activations=False):
        for conv_count in range(len(self.conv_layers)):
            x = self.conv_layers[conv_count](x)
            if get_activations:
                self.activations[conv_count].append(x)
            x = F.relu(x)
            if conv_count %2 ==1:
                x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        # get loss
        loss = self.cross_entropy_loss(logits, y)
        # get acc
        self.train_acc(logits, y)
        # log loss and acc
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc)
        batch_dictionary={
	            "train_loss": loss, "train_acc": self.train_acc, 'loss': loss
	        }
        return batch_dictionary

    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        
        self.log('train_loss_epoch', avg_loss, sync_dist=True)
        self.log('train_acc_epoch', self.train_acc, sync_dist=True)
        gc.collect()
    
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            x, y = val_batch
            logits = self.forward(x, get_activations=True)
            # get loss
            loss = self.cross_entropy_loss(logits, y)
            # get acc
            self.valid_acc(logits, y)
            # get class acc
            # class_acc = {}
            # if self.classnames == None:
            #     self.classnames = list(set(y))
            # corr_pred = {classname: 0 for classname in self.classnames}
            # total_pred = {classname: 0 for classname in self.classnames}
            # for label, prediction in zip(y, labels_hat):
            #         if label == prediction:
            #             corr_pred[self.classnames[label]] += 1
            #         total_pred[self.classnames[label]] += 1
            # for classname, correct_count in corr_pred.items():
            #     accuracy = 0
            #     if correct_count != 0 and total_pred[classname] != 0:
            #         accuracy = 100 * float(correct_count) / total_pred[classname]
            #     class_acc[classname] = accuracy
            # get novelty score
            novelty_score = self.compute_feature_novelty()

            # log loss, acc, class acc, and novelty score
            # clear out activations
            for i in range(len(self.conv_layers)):
                self.activations[i] = []
            self.log('val_loss', loss)
            self.log('val_acc', self.valid_acc)
            # self.log('val_class_acc', class_acc)
            self.log('val_novelty', novelty_score)
            batch_dictionary = {'val_loss': loss, 
                                'val_acc': self.valid_acc, 
                                # 'val_class_acc': class_acc, 
                                'val_novelty': novelty_score 
                                }
        return batch_dictionary
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        # avg_class_acc = {}
        # for x in outputs:
        #     for k, v in x['val_class_acc'].items():
        #         avg_class_acc[k] = v
        avg_novelty = np.stack([x['val_novelty'] for x in outputs]).mean()
        self.avg_novelty = avg_novelty
        self.log('val_loss_epoch', avg_loss, sync_dist=True)
        self.log('val_acc_epoch', self.valid_acc, sync_dist=True)
        # self.log('val_class_acc_epoch', avg_class_acc)
        self.log('val_novelty_epoch', avg_novelty, sync_dist=True)
        gc.collect()

    def get_fitness(self, batch):
        with torch.no_grad():
            x, y = batch
            logits = self.forward(x, get_activations=True)
            novelty_score = self.compute_feature_novelty()
            # clear out activations
            for i in range(len(self.conv_layers)):
                self.activations[i] = []
        return novelty_score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)
        
    def set_filters(self, filters):
        for i in range(len(filters)):
            self.conv_layers[i].weight.data = filters[i]
    
    def get_filters(self, numpy=False):
        if numpy:
            return [m.weight.data.detach().cpu().numpy() for m in self.conv_layers]
        return [m.weight.data.detach().cpu() for m in self.conv_layers]

    def get_features(self, numpy=False):
        if numpy:
            return [self.activations[a][0] for a in range(len(self.activations))]
        return [self.activations[a][0] for a in range(len(self.activations))]
    
    def compute_activation_dist(self):
        activations = self.get_features(numpy=True)
        return helper.get_dist(activations)
    
    def compute_weight_dist(self):
        weights = self.get_filters(True)
        return helper.get_dist(weights)

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
            if self.diversity['type']=='relative':
                l.append(helper.diversity_relative(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
            elif self.diversity['type']=='original':
                l.append(helper.diversity_orig(self.activations[i], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
            elif self.diversity['type']=='absolute':
                l.append(helper.diversity(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
            elif self.diversity['type']=='cosine':
                l.append(helper.diversity_cosine_distance(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
            elif self.diversity['type'] == 'constant':
                l.append(helper.diversity_constant(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
            else:
                l.append(helper.diversity(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))

        if self.diversity['ldop'] == 'sum':
            return(sum(l))
        elif self.diversity['ldop'] == 'mean':
            return(np.mean(l))
        elif self.diversity['ldop'] == 'w_mean':
            total_channels = 0
            for i in range(len(self.conv_layers)):
                total_channels+=self.conv_layers[i].out_channels
            return(np.sum([l[i]*(self.conv_layers[i].out_channels)/total_channels for i in range(len(l))]))


# class diversityMetric(torchmetrics.Metric):
#     def __init__(self):
#         super().__init__()
#         self.add_state("diversity_score", default=torch.tensor(0))

#     def update(self, diversity_score):
#         self.diversity_score = diversity_score

#     def compute(self):
#         return self.diversity_score