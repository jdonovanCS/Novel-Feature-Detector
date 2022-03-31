from distutils.command.config import config
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import randomdataset as rd
import matplotlib.pyplot as plt
import scikits.bootstrap as bootstrap
import warnings
warnings.filterwarnings('ignore')
import wandb
import pl_bolts.datamodules
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from net import Net
from cifar100datamodule import CIFAR100DataModule

def create_random_images(num_images=200):
    paths = []
    for i in range(num_images):
        rgb = np.random.randint(255, size=(32,32,3), dtype=np.uint8)
        cv2.imwrite('images/{}.png'.format(i), rgb)
        paths.append('images/{}.png'.format(i))
    return paths

def load_random_images(random_image_paths, batch_size=64):
    train_dataset = rd.RandomDataset(random_image_paths)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader

def train_network(data_module, filters=None, epochs=2, save_path=None, fixed_conv=False, val_interval=1, novelty_interval=None):
    net = Net(num_classes=data_module.num_classes, classnames=list(data_module.dataset_test.classes))
    net = net.to(device)
    print(net.device)
    if filters is not None:
        for i in range(len(net.conv_layers)):
            net.conv_layers[i].weight.data = torch.tensor(filters[i])
            if fixed_conv:
                for param in net.conv_layers[i].parameters():
                    param.requires_grad = False

    if save_path is None:
        save_path = PATH
    wandb_logger = WandbLogger(log_model=True)
    trainer = pl.Trainer(max_epochs=epochs, default_root_dir=save_path, logger=wandb_logger, check_val_every_n_epoch=val_interval, accelerator="gpu")
    wandb_logger.watch(net, log="all")
    trainer.fit(net, data_module)
    wandb_logger.unwatch(net)

    # torch.save(net.state_dict(), save_path)
    # return record_progress

def get_data_module(dataset, batch_size):
    match dataset.lower():
        case 'cifar10' | 'cifar-10':
            data_module = pl_bolts.datamodules.CIFAR10DataModule(batch_size=batch_size, data_dir="data/")
        case 'cifar100' | 'cifar-100':
            data_module = CIFAR100DataModule(batch_size=batch_size, data_dir="data/")
        case _:
            print('Please supply dataset of CIFAR-10 or CIFAR-100')
            exit()
    return data_module

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
    
def log(input):
    wandb.log(input)

def update_config():
    wandb.config = config

def run(seed=True):
    torch.multiprocessing.freeze_support()
    if seed:
        pl.seed_everything(42, workers=True)
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global PATH
    PATH = './cifar_net.pth'
    global config
    wandb.init(project="novel-feature-detectors")


if __name__ == '__main__':
    run()