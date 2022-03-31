import gc
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import helper_hpc as helper

# DEFINE a CONV NN

class Net(pl.LightningModule):
    def __init__(self, num_classes=10, classnames=None):
        super().__init__()
        self.save_hyperparameters()
        self.BatchNorm1 = nn.BatchNorm2d(32)
        self.BatchNorm2 = nn.BatchNorm2d(128)
        self.BatchNorm3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout1 = nn.Dropout2d(0.05)
        self.dropout2 = nn.Dropout2d(0.1)
        self.conv_layers = [nn.Conv2d(3, 32, 3, padding=1), 
                            nn.Conv2d(32, 64, 3, padding=1), 
                            nn.Conv2d(64, 128, 3, padding=1), 
                            nn.Conv2d(128, 128, 3, padding=1), 
                            nn.Conv2d(128, 256, 3, padding=1), 
                            nn.Conv2d(256, 256, 3, padding=1)]
        self.activations = {}
        for i in range(len(self.conv_layers)):
            self.activations[i] = []

        self.classnames = classnames

    def forward(self, x):
        conv_count = 0
        x = self.conv_layers[conv_count](x)
        self.activations[conv_count].append(x)
        conv_count += 1
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = self.conv_layers[conv_count](x)
        self.activations[conv_count].append(x)
        conv_count += 1
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv_layers[conv_count](x)
        self.activations[conv_count].append(x)
        conv_count += 1
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = self.conv_layers[conv_count](x)
        self.activations[conv_count].append(x)
        conv_count += 1
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.conv_layers[conv_count](x)
        self.activations[conv_count].append(x)
        conv_count += 1
        x = self.BatchNorm3(x)
        x = F.relu(x)
        x = self.conv_layers[conv_count](x)
        self.activations[conv_count].append(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
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
        labels_hat = torch.argmax(logits, 1)
        acc = torch.sum(y==labels_hat).item()/(len(y)*1.0)
        # log loss and acc
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        batch_dictionary={
	            "train_loss": loss, "train_acc": acc
	        }
        return batch_dictionary

    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        
        self.log('train_loss_epoch', avg_loss)
        self.log('train_acc_epoch', avg_acc)
        gc.collect()
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        # get loss
        loss = self.cross_entropy_loss(logits, y)
        # get acc
        labels_hat = torch.argmax(logits, 1)
        acc = torch.sum(y==labels_hat).item()/(len(y)*1.0)
        # get class acc
        class_acc = {}
        if self.classnames == None:
            classnames = list(set(y))
        corr_pred = {classname: 0 for classname in classnames}
        total_pred = {classname: 0 for classname in classnames}
        for label, prediction in zip(y, labels_hat):
                if label == prediction:
                    corr_pred[classnames[label]] += 1
                total_pred[classnames[label]] += 1
        for classname, correct_count in corr_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            class_acc[classname] = accuracy
        # get novelty score
        novelty_score = self.compute_feature_novelty()
        # clear out activations
        for i in range(len(self.conv_layers)):
            self.activations[i] = []
        # log loss, acc, class acc, and novelty score
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_class_acc', class_acc)
        self.log('val_novelty', novelty_score)
        batch_dictionary = {'val_loss': loss, 'val_acc': acc, 'val_class_acc': class_acc, 'val_novelty': novelty_score}
        return batch_dictionary
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_class_acc = torch.stack([x['val_class_acc'] for x in outputs]).mean()
        avg_novelty = torch.stack([x['val_novelty'] for x in outputs]).mean()
        self.log('val_loss_epoch', avg_loss)
        self.log('val_acc_epoch', avg_acc)
        self.log('val_class_acc_epoch', avg_class_acc)
        self.log('val_novelty_epoch', avg_novelty)
        gc.collect()

    def get_fitness(self, batch):
        x, y = batch
        logits = self.forward(x)
        novelty_score = self.compute_feature_novelty()
        # clear out activations
        for i in range(len(self.conv_layers)):
            self.activations[i] = []
        return novelty_score

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        # get loss
        loss = self.cross_entropy_loss(logits, y)
        # get acc
        labels_hat = torch.argmax(logits, 1)
        acc = torch.sum(y==labels_hat).item()/(len(y)*1.0)
        # get class acc
        class_acc = {}
        classes = list(set(y.detach().numpy()))
        corr_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        for label, prediction in zip(y.detach().numpy(), labels_hat.detach().numpy()):
            if label == prediction:
                corr_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
        for classname, correct_count in corr_pred.items():
            accuracy = None
            if total_pred[classname] != 0:
                accuracy = 100 * float(correct_count) / total_pred[classname]
            class_acc[classname] = accuracy
        # get novelty score
        novelty_score = self.compute_feature_novelty()
        # clear out activations
        for i in range(len(self.conv_layers)):
            self.activations[i] = []
        # log loss, acc, class acc, and novelty score
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_class_acc', class_acc)
        self.log('test_novelty', novelty_score)
        batch_dictionary = {'test_loss': loss, 'test_acc': acc, 'test_class_acc': class_acc, 'test_novelty': novelty_score}
        return batch_dictionary

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_class_acc = torch.stack([x['test_class_acc'] for x in outputs]).mean()
        avg_novelty = torch.stack([x['test_novelty'] for x in outputs]).mean()
        self.log('test_loss_epoch', avg_loss)
        self.log('test_acc_epoch', avg_acc)
        self.log('test_class_acc_epoch', avg_class_acc)
        self.log('test_novelty_epoch', avg_novelty)
        gc.collect()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
        
    def set_filters(self, filters):
        for i in range(len(filters)):
            self.conv_layers[i].weight.data = filters[i]
    
    def get_filters(self):
        return [m.weight.data for m in self.conv_layers]

    def compute_feature_novelty(self):
        dist = []
        avg_dist = {}
        # for each conv layer
        for layer in self.activations:
            dist = []
            # for each activation 3d(batch, h, w)
            for batch in self.activations[layer]:
                # for each activation
                for ind_activation in batch:
                    
                    for ind_activation2 in batch:
                        dist.append(np.abs(ind_activation.detach().numpy(), ind_activation2.detach().numpy()))
            avg_dist[str(layer)] = np.mean((dist))
        return(sum(avg_dist.values()))
