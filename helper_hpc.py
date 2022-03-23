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


# Need to separate this file into functions and classes

def load_CIFAR_10(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = batch_size

    # Dowload training data
    trainset = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, testset, trainloader, testloader, classes

def load_CIFAR_100(batch_size=64):
    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    batch_size = batch_size

    # Dowload training data
    trainset = torchvision.datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=train_transform
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=test_transform
    )

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = tuple(trainset.classes)

    return trainset, testset, trainloader, testloader, classes

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

class Net(nn.Module):
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
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.conv5(x)
        x = self.BatchNorm3(x)
        x = F.relu(x)
        x = self.conv6(x)
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

def get_activations(trainloader, filters, num_ims_used=64):
    net = Net()
    for i in range (len(net.conv_layers)):
        net.conv_layers[i].weight.data = filters[i]
    net = net.to(device)

    activations = {}
    def get_features(name):
        def hook(model, input, output):
            if name not in activations.keys():
                activations[name] = []
            activations[name].append(output.cpu().detach().numpy())
        return hook

    for i in range (len(net.conv_layers)):
        net.conv_layers[i].register_forward_hook(get_features(i))

    #TODO: The below code is far too slow
    total = 0
    while total < num_ims_used:
        data = next(iter(trainloader))
        inputs, labels = data
        # inputs = np.transpose(inputs, (0, 3, 2, 1)).float()
        inputs = inputs.float()
        inputs = inputs.to(device)
        # labels = labels.to(device)
        outputs = net(inputs)
        
        total += np.array(labels).size
    # print(activations)
    return activations

def get_random_filters():
    net = Net().to(device)

    filters = []
    for i in range(len(net.conv_layers)):
        filters.append(net.conv_layers[i].weight.data.cpu().detach())
    return np.array(filters)


def train_network(trainloader, testloader, classes, filters=None, epochs=2, save_path=None, fixed_conv=False, novelty_interval=0):
    net = Net(num_classes=len(list(classes)))
    if filters is not None:
        for i in range(len(net.conv_layers)):
            net.conv_layers[i].weight.data = torch.tensor(filters[i])
            if fixed_conv:
                for param in net.conv_layers[i].parameters():
                    param.requires_grad = False

    net = net.to(device)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    record_progress = {}
    record_progress['running_loss'] = []
    record_progress['running_acc'] = []
    record_progress['novelty_score'] = []

    # Load all images and labels into memory, then send to device instead of loading by batch from drive->mem->device.
    epochs = epochs
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0.0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the param gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i + 1, running_loss/2000))
                record_progress['running_loss'].append({'epoch': epoch+1, 'iter': i+1, 'running_loss': running_loss/2000})
                running_loss = 0.0

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted==labels).sum().item()
            total += labels.size(0)
        accuracy = 100 * correct / total
        print('Accuracy of the network on training set at epoch %d: %d %%' % (epoch+1, accuracy))
        record_progress['running_acc'].append({'epoch': epoch+1, 'accuracy': accuracy})
        import evolution as evol
        if not fixed_conv and novelty_interval != 0 and epoch % novelty_interval == 0:
            trained_filters = []
            for j in range(len(net.conv_layers)):
                trained_filters.append(net.conv_layers[j].weight.data)
            activations = get_activations(trainloader, trained_filters)
            novelty_score = evol.compute_feature_novelty(activations)
            record_progress['novelty_score'].append({'epoch': epoch+1, 'novelty': novelty_score})
        # run to compare the accuracy of network on test set.
        # if save_path is None:
        #     save_path = PATH
        # torch.save(net.state_dict(), save_path)
        # assess_accuracy(testloader, classes, save_path)

    print('Finished Training')
    if save_path is None:
        save_path = PATH

    torch.save(net.state_dict(), save_path)
    return record_progress

def assess_accuracy(testloader, classes, save_path=None):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    print("Ground Truth: ", " ".join('%5s' % classes[labels[j]] for j in range(4)))
    
    net = Net(num_classes=len(list(classes)))
    if save_path is None:
        save_path = PATH
    net.load_state_dict(torch.load(save_path))
    net = net.to(device)
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)
    record_accuracy = {}

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct/total))
    record_accuracy['overall'] = (100 * correct/total)

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
        record_accuracy[classname] = accuracy

    return record_accuracy

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
    

def run(batch_size_input=64):
    torch.multiprocessing.freeze_support()

    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Device:', device)
    global PATH
    PATH = './cifar_net.pth'
    global batch_size
    batch_size = batch_size_input


    
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