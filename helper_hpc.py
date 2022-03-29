import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import randomdataset as rd
import matplotlib.pyplot as plt
import scikits.bootstrap as bootstrap
import warnings
warnings.filterwarnings('ignore')
import wandb
import pytorch_lighning as pl
import pl_bolts.datamodules
import evolution as evol
from pytorch_lightning_loggers import WandbLogger

# Need to separate this file into functions and classes

# def load_CIFAR_10(batch_size=64):
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     batch_size = batch_size

#     # Dowload training data
#     trainset = torchvision.datasets.CIFAR10(
#         root="data",
#         train=True,
#         download=True,
#         transform=transform
#     )

#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

#     testset = torchvision.datasets.CIFAR10(
#         root="data",
#         train=False,
#         download=True,
#         transform=transform
#     )

#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#     return trainset, testset, trainloader, testloader, classes

# def load_CIFAR_100(batch_size=64):
#     stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
#     train_transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
#         transforms.ToTensor(),
#         transforms.Normalize(*stats)
#     ])

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(*stats)
#     ])

#     batch_size = batch_size

#     # Dowload training data
#     trainset = torchvision.datasets.CIFAR100(
#         root="data",
#         train=True,
#         download=True,
#         transform=train_transform
#     )

#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

#     testset = torchvision.datasets.CIFAR100(
#         root="data",
#         train=False,
#         download=True,
#         transform=test_transform
#     )

#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

#     classes = tuple(trainset.classes)

#     return trainset, testset, trainloader, testloader, classes

def create_random_images(num_images=200):
    paths = []
    for i in range(num_images):
        rgb = np.random.randint(255, size=(32,32,3), dtype=np.uint8)
        cv2.imwrite('images/{}.png'.format(i), rgb)
        paths.append('images/{}.png'.format(i))
    return paths

def load_random_images(random_image_paths):
    train_dataset = rd.RandomDataset(random_image_paths)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader

# Show some training images
def imshow(img):
    print(img.shape)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def load_and_show_some_images(trainloader, classes, batch_size):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range (batch_size)))


# DEFINE a CONV NN

class Net(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.BatchNorm3 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        # self.fc3 = nn.Linear(84, 10)
        self.dropout1 = nn.Dropout2d(0.05)
        self.dropout2 = nn.Dropout2d(0.1)
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]
        # print(self.conv1.weight.shape)

    def forward(self, x):
        x = self.conv1(x)
        self.conv1_act = x
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        self.conv2_act = x
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        self.conv3_act = x
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = self.conv4(x)
        self.conv4_act = x
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.conv5(x)
        self.conv5_act = x
        x = self.BatchNorm3(x)
        x = F.relu(x)
        x = self.conv6(x)
        self.conv6_act = x
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
        classes = list(set(y))
        corr_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        for label, prediction in zip(y, labels_hat):
                if label == prediction:
                    corr_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
        for classname, correct_count in corr_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            class_acc[classname] = accuracy
        # get novelty score
        # if not fixed_conv and novelty_interval != 0 and epoch % novelty_interval == 0:
        trained_filters = []
        for j in range(len(self.conv_layers)):
            trained_filters.append(self.conv_layers[j].weight.data)
            # print(trained_filters[j])
        self.activations = {}
        for i in range(len(self.conv_layers)):
            self.activations[i] = []
        for i in range(len(x)):
            x_act = self.get_activations(x[i])
            for j in range(len(x_act)):
                self.activations[j].append(x_act[j])
        novelty_score = evol.compute_feature_novelty(activations)
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
        classes = list(set(y))
        corr_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        for label, prediction in zip(y, labels_hat):
                if label == prediction:
                    corr_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
        for classname, correct_count in corr_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            class_acc[classname] = accuracy
        # get novelty score
        # if not fixed_conv and novelty_interval != 0 and epoch % novelty_interval == 0:
        trained_filters = []
        for j in range(len(self.conv_layers)):
            trained_filters.append(self.conv_layers[j].weight.data)
            # print(trained_filters[j])
        self.activations = {}
        for i in range(len(self.conv_layers)):
            self.activations[i] = []
        for i in range(len(x)):
            x_act = self.get_activations(x[i])
            for j in range(len(x_act)):
                self.activations[j].append(x_act[j])
        novelty_score = evol.compute_feature_novelty(activations)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3, momentum=0.9)
        return optimizer

    def get_activations(self, x):
        return[self.conv1_act, self.conv2_act, self.conv3_act, self.conv4_act, self.conv5_act, self.conv6_act]

    def set_filters(self, filters):
        for i in range(len(filters)):
            self.conv_layers[i].weight.data = filters[i]

# def get_activations(trainloader, filters, num_ims_used=64):
#     net = Net()
#     for i in range (len(net.conv_layers)):
#         net.conv_layers[i].weight.data = filters[i]
#     net = net.to(device)

#     activations = {}
#     def get_features(name):
#         def hook(model, input, output):
#             if name not in activations.keys():
#                 activations[name] = []
#             activations[name].append(output.cpu().detach().numpy())
#         return hook

#     for i in range (len(net.conv_layers)):
#         net.conv_layers[i].register_forward_hook(get_features(i))

#     #TODO: The below code is far too slow
#     total = 0
#     while total < num_ims_used:
#         data = next(iter(trainloader))
#         inputs, labels = data
#         # inputs = np.transpose(inputs, (0, 3, 2, 1)).float()
#         inputs = inputs.float()
#         inputs = inputs.to(device)
#         # labels = labels.to(device)
#         outputs = net(inputs)
        
#         total += np.array(labels).size
#     # print(activations)
#     return activations

def get_random_filters():
    net = Net().to(device)

    filters = []
    for i in range(len(net.conv_layers)):
        filters.append(net.conv_layers[i].weight.data.cpu().detach())
    return np.array(filters)


def train_network(data_module, filters=None, epochs=2, save_path=None, fixed_conv=False, val_interval=1, novelty_interval=None):
    net = Net(num_classes=data_module.num_classes)
    if filters is not None:
        for i in range(len(net.conv_layers)):
            net.conv_layers[i].weight.data = torch.tensor(filters[i])
            if fixed_conv:
                for param in net.conv_layers[i].parameters():
                    param.requires_grad = False

    if save_path is None:
        save_path = PATH
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=epochs, default_root_dir=save_path, logger=wandb_logger)
    # data_module = CIFAR10DataModule()
    trainer.fit(net, data_module, check_val_every_n_epoch=val_interval)

    # torch.save(net.state_dict(), save_path)
    # return record_progress

def get_data_module(dataset, batch_size):
    match dataset.lower():
        case 'cifar10' | 'cifar-10':
            data_module = pl_bolts.datamodules.CIFAR10DataModule(batch_size=batch_size)
        case 'cifar100' | 'cifar-100':
            data_module = pl_bolts.datamodules.CIFAR100DataModule(batch_size=batch_size)
        case _:
            print('Please supply dataset of CIFAR-10 or CIFAR-100')
            exit()
    return data_module


# def assess_accuracy(testloader, classes, save_path=None):
    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    # # imshow(torchvision.utils.make_grid(images))
    # print("Ground Truth: ", " ".join('%5s' % classes[labels[j]] for j in range(4)))
    
    # net = Net(num_classes=len(list(classes)))
    # if save_path is None:
    #     save_path = PATH
    # net.load_from_checkpoint(save_path)
    # net = net.to(device)
    # outputs = net(images.to(device))
    # _, predicted = torch.max(outputs, 1)
    # record_accuracy = {}

    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    
    # correct = 0
    # total = 0

    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         # images, labels = data[0].to(device), data[1].to(device)
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted==labels).sum().item()
    # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct/total))
    # record_accuracy['overall'] = (100 * correct/total)

    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}

    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         outputs = net(images)
    #         _, predictions = torch.max(outputs, 1)
    #         for label, prediction in zip(labels, predictions):
    #             if label == prediction:
    #                 correct_pred[classes[label]] += 1
    #             total_pred[classes[label]] += 1

    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
    #     record_accuracy[classname] = accuracy

    # return record_accuracy

def plot_mean_and_bootstrapped_ci_multiple(input_data = None, title = 'overall', name = "change this", x_label = "x", y_label = "y", save_name="", compute_CI=True, maximum_possible=None, show=None, sample_interval=None):
    """ 
     
    parameters:  
    input_data: (numpy array of numpy arrays of shape (max_k, num_repitions)) solution met
    name: numpy array of string names for legend 
    x_label: (string) x axis label 
    y_label: (string) y axis label 
     
    returns: 
    None 
    """ 
 
    generations = len(input_data[0])
 
    fig, ax = plt.subplots() 
    ax.set_xlabel(x_label) 
    ax.set_ylabel(y_label) 
    ax.set_title(title) 
    for i in range(len(input_data)): 
        CIs = [] 
        mean_values = [] 
        for j in range(generations): 
            mean_values.append(np.mean(input_data[i][j])) 
            if compute_CI:
                CIs.append(bootstrap.ci(input_data[i][j], statfunction=np.mean)) 
        mean_values=np.array(mean_values) 
 
        high = [] 
        low = [] 
        if compute_CI:
            for j in range(len(CIs)): 
                low.append(CIs[j][0]) 
                high.append(CIs[j][1]) 
 
        low = np.array(low) 
        high = np.array(high) 

        y = range(0, generations)
        if (sample_interval != None):
            y = np.array(y)*sample_interval 
        ax.plot(y, mean_values, label=name[i])
        if compute_CI:
            ax.fill_between(y, high, low, alpha=.2) 
        ax.legend()
    
    if maximum_possible:
        ax.hlines(y=maximum_possible, xmin=0, xmax=generations, linewidth=2, color='r', linestyle='--', label='best poss. acc.')
        ax.legend()

    if save_name != "":
        plt.savefig('plots/' + save_name)
    if show != None:
        plt.show()
    

def run():
    torch.multiprocessing.freeze_support()
    pl.seed_everything(42, workers=True)

    # global device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print('Device:', device)
    global PATH
    PATH = './cifar_net.pth'
    # global batch_size
    # batch_size = batch_size_input
    wandb.init(project="novel-feature-detectors")


    
    # trainset, testset, trainloader, testloader, classes = load_CIFAR_10(batch_size)
    # load_and_show_some_images(trainloader, classes, batch_size)

    # train_network_on_CIFAR_10(trainloader)
    # assess_accuracy(testloader, classes)

    # random_image_paths = create_random_images()
    # trainloader = load_random_images(random_image_paths)
    # filters = get_random_filters()
    # for f in filters:
    #     print(f.shape)
    # activations = get_activations(trainloader, filters)
    # for a in activations:
    #     print(a.shape)


if __name__ == '__main__':
    run()