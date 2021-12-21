import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import randomdataset as rd

# Need to separate this file into functions and classes

def load_CIFAR_10(batch_size=4):
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

def create_random_images(num_images=200):
    paths = []
    for i in range(num_images):
        rgb = np.random.randint(255, size=(32,32,3), dtype=np.uint8)
        cv2.imwrite('images/{}.png'.format(i), rgb)
        paths.append('images/{}.png'.format(i))
    return paths

def load_random_images(random_image_paths):
    train_dataset = rd.RandomDataset(random_image_paths)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        print(self.conv1.weight.shape)
        print(self.conv2.weight.shape)
        print(self.conv3.weight.shape)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(21632, 512)
        self.fc2 = nn.Linear(512, 10)
        # self.fc3 = nn.Linear(84, 10)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

        output = F.log_softmax(x, dim=1)
        return output

def get_activations(trainloader):
    net = Net().to(device)

    global view_conv1
    def hook_conv1_fn(module, input, output):
        global view_conv1
        view_conv1 = output
    global view_conv2
    def hook_conv2_fn(module, input, output):
        global view_conv2
        view_conv2 = output
    global view_conv3
    def hook_conv3_fn(module, input, output):
        global view_conv3
        view_conv3 = output

    hook_conv1 = net.conv1.register_forward_hook(hook_conv1_fn)
    hook_conv2 = net.conv2.register_forward_hook(hook_conv2_fn)
    hook_conv3 = net.conv3.register_forward_hook(hook_conv3_fn)

    activations = {"conv1": [], "conv2": [], "conv3": []}
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = np.transpose(inputs, (0, 3, 2, 1)).float()
        inputs = inputs.to(device)
        # labels = labels.to(device)
        outputs = net(inputs)
        activations['conv1'].append(view_conv1.cpu().detach().numpy())
        activations['conv2'].append(view_conv2.cpu().detach().numpy())
        activations['conv3'].append(view_conv3.cpu().detach().numpy())
    
    return activations

def train_network_on_CIFAR_10(trainloader):
    net = Net().to(device)
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = 2
    for epoch in range(epochs):
        running_loss = 0.0
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
                running_loss = 0.0
    print('Finished Training')
    torch.save(net.state_dict(), PATH)

def assess_accuracy(testloader, classes):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print("Ground Truth: ", " ".join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net().to(device)
    net.load_state_dict(torch.load(PATH))
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct/total))

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


def run():
    torch.multiprocessing.freeze_support()

    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Device:', device)
    global PATH
    PATH = './cifar_net.pth'
    global batch_size
    batch_size = 4


    
    trainset, testset, trainloader, testloader, classes = load_CIFAR_10(batch_size)
    load_and_show_some_images(trainloader, classes, batch_size)

    # train_network_on_CIFAR_10(trainloader)
    # assess_accuracy(testloader, classes)

    random_image_paths = create_random_images()
    trainloader = load_random_images(random_image_paths)
    activations = get_activations(trainloader)
    for k, v in activations.items():
        print(np.array(v).shape)


if __name__ == '__main__':
    run()