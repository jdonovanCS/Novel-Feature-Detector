import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models


class Net(pl.LightningModule):
    def __init__(self, num_classes=10, classnames=None, diversity=None, lr=5e-5, bn=True):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes

        self.classnames = classnames
        self.diversity = diversity
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = lr

        if bn:
            self.model = models.vgg16_bn(pretrained=False, num_classes=self.num_classes)
        else:
            self.model = models.vgg16(pretrained=False, num_classes=self.num_classes)
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=self.num_classes)

        # self.activations = {}
        # for i in range(len(self.conv_layers)):
        #     self.activations[i] = []


    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        logits = self.forward(x)

        loss = self.cross_entropy_loss(logits, y)
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
    
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            x, y = val_batch
            
            logits = self.forward(x)

            loss = self.cross_entropy_loss(logits, y)

            self.valid_acc(logits, y)

            self.log('val_loss', loss)
            self.log('val_acc', self.valid_acc)
            batch_dictionary = {'val_loss': loss, 
                                'val_acc': self.valid_acc
                                }
        return batch_dictionary
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', avg_loss, sync_dist=True)
        self.log('val_acc_epoch', self.valid_acc, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        # return F.nll_loss(logits, labels)
        return F.cross_entropy(logits, labels)
    
    # def get_fitness(self, batch):
    #     with torch.no_grad():
    #         x, y = batch
    #         logits = self.forward(x, get_activations=True)
    #         novelty_score = self.compute_feature_novelty()
    #         # clear out activations
    #         for i in range(len(self.conv_layers)):
    #             self.activations[i] = []
    #     return novelty_score

    def set_filters(self, filters):
        count = 0
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Conv2d)):
                z = torch.tensor(filters[count])
                z = z.type_as(m.weight.data)
                m.weight.data = z
                count += 1
    
    # def get_filters(self, numpy=False):
    #     if numpy:
    #         return [m.weight.data.detach().cpu().numpy() for m in self.conv_layers]
    #     return [m.weight.data.detach().cpu() for m in self.conv_layers]

    # def get_features(self, numpy=False):
    #     if numpy:
    #         return [self.activations[a][0] for a in range(len(self.activations))]
    #     return [self.activations[a][0] for a in range(len(self.activations))]
    
    # def compute_activation_dist(self):
    #     activations = self.get_features(numpy=True)
    #     return helper.get_dist(activations)
    
    # def compute_weight_dist(self):
    #     weights = self.get_filters(True)
    #     return helper.get_dist(weights)

    # def compute_feature_novelty(self):
        
    #     # start = time.time()
    #     # layer_totals = {}
    #     # with torch.no_grad():
    #     #     # for each conv layer 4d (batch, channel, h, w)
    #     #     for layer in range(len(self.activations)):
    #     #         B = len(self.activations[layer][0])
    #     #         C = len(self.activations[layer][0][0])
    #     #         a = self.activations[layer][0]
    #     #         layer_totals[layer] = torch.abs(a.unsqueeze(2) - a.unsqueeze(1)).sum().item()
    #     # end = time.time()
    #     # print('gpu answer: {}'.format(sum(layer_totals.values())))
    #     # print('gpu time: {}'.format(end-start))
    #     # return(sum(layer_totals.values()))

    #         # layer_totals[layer] = np.abs(np.expand_dims(a, axis=2) - np.expand_dims(a, axis=1)).sum().item()

    #     if self.diversity == None:
    #         return 0
    #     l = []
    #     for i in self.activations:
    #         print(len(self.activations[i]))
    #         if len(self.activations[i]) == 0:
    #             continue
    #         if type(self.activations[i][0]) == torch.Tensor:
    #             self.activations[i][0] = self.activations[i][0].detach().cpu().numpy()
    #         if self.diversity['type']=='relative':
    #             l.append(helper.diversity_relative(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
    #         elif self.diversity['type']=='original':
    #             l.append(helper.diversity_orig(self.activations[i], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
    #         elif self.diversity['type']=='absolute':
    #             l.append(helper.diversity(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
    #         elif self.diversity['type']=='cosine':
    #             l.append(helper.diversity_cosine_distance(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
    #         elif self.diversity['type'] == 'constant':
    #             l.append(helper.diversity_constant(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))
    #         else:
    #             l.append(helper.diversity(self.activations[i][0], self.diversity['pdop'], self.diversity['k'], self.diversity['k_strat']))

    #     if self.diversity['ldop'] == 'sum':
    #         return(sum(l))
    #     elif self.diversity['ldop'] == 'mean':
    #         return(np.mean(l))
    #     elif self.diversity['ldop'] == 'w_mean':
    #         total_channels = 0
    #         for i in range(len(self.conv_layers)):
    #             total_channels+=self.conv_layers[i].out_channels
    #         return(np.sum([l[i]*(self.conv_layers[i].out_channels)/total_channels for i in range(len(l))]))
