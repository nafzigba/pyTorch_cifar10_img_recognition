from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Preparing for Data
print('==> Preparing data..')

# Training Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Testing Data preparation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


#nn.BatchNorm1d(120)
#nn.Dropout(p=0.5)
#max_pool2d(out, 2)
#.view(out.size(0), -1)
#nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
#nn.Tanh(),
from __future__ import print_function
import torch  # import the torch library
import torch.nn as nn # use the nn module (class)
import torch.nn.functional as F    # use the nn module as function
import torch.optim as optim # optimization (i.e., SGD, ada,)
import torchvision # load the dataset
import torchvision.transforms as transforms # adjust the input image
import time # check the processing overhead

# Preparing for Data
print('==> Preparing data..')

# Training Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Testing Data preparation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# total number of classes
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the class for LeNet
class LeNet(nn.Module):
    def __init__(self):# will be called when you create an object
        super(LeNet, self).__init__()

        #torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding)
        #kernel_size: _size_2_t, stride: _size_2_t | None = None, padding: _size_2_t = 0, ceil_mode: bool = False,
        # count_include_pad: bool = True, divisor_override: int | None = None) -> None

        '''
        Maximum number of layers (convolution and fully-connected) of your proposed model
        should be 10.

        I double checked to make sure BatchNormalization didn't count as a layer so I could use them freely:
        I looked on StackOverflow and the consensus was that they did not count.
        https://stackoverflow.com/questions/65163196/does-batchnormalization-count-as-a-layer-in-a-network
        "
        In DeepLearning literature, an X layer network simply refers to the usage of learnable layers that constitute the representational capacity of the network.
        Activation layers, normalization layers (such as NLR, BatchNorm, etc), Downsampling layers (such as Maxpooling, etc) are not considered.
        Layers such as CNN, RNN, FC, and the likes that are responsible for the representational capacity of the network are counted.
        "
        '''

        self.c1 = nn.Conv2d(3,8,13,1,7)
        self.bn1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8,16,9,1,5)
        self.bn2 = nn.BatchNorm2d(16)
        self.c3 = nn.Conv2d(16,32,7,1,3)
        self.bn3 = nn.BatchNorm2d(32)
        self.c4 = nn.Conv2d(32,64,5,1,2)
        self.bn4 = nn.BatchNorm2d(64)
        self.cmid = nn.Conv2d(64,128,3,1)

        self.mp1 = nn.MaxPool2d(2,2)
        self.c5 = nn.Conv2d(128,128,7,1)
        self.bn5 = nn.BatchNorm2d(128)

        self.mp2 = nn.MaxPool2d(2,2)
        self.c6 = nn.Conv2d(128,320,5,1)
        self.bn6 = nn.BatchNorm2d(320)

        self.ft = nn.Flatten()

        self.F6 = nn.Linear(320, 160)
        self.bn7 = nn.BatchNorm1d(160)
        self.F7 = nn.Linear(160, 80)
        self.bn8 = nn.BatchNorm1d(80)
        self.F8 = nn.Linear(80, 10)

        self.MyRelu=nn.ReLU()




    def forward(self, x):  # this function will be called when you run the model

        x=self.c1(x)
        x=self.bn1(x)
        x=self.MyRelu(x)

        x=self.c2(x)
        x=self.bn2(x)
        x=F.tanh(x)

        x=self.c3(x)
        x=self.bn3(x)
        x=self.MyRelu(x)

        x=self.c4(x)
        x=self.bn4(x)
        x=F.tanh(x)

        x=self.cmid(x)
        x=F.tanh(x)

        x=self.mp1(x)
        x=self.c5(x)
        x=self.bn5(x)

        x=self.mp2(x)
        x=self.c6(x)
        x=self.bn6(x)

        x=self.MyRelu(x)
        x=self.ft(x)
        x=self.F6(x)
        x=self.bn7(x)
        x=self.MyRelu(x)
        x=self.F7(x)
        x=self.bn8(x)
        x=self.MyRelu(x)
        x=self.F8(x)

        out = F.softmax(x,dim=0)

        return x



def train(model, device, train_loader, optimizer, epoch):
    model.train() # set the model into training model (evaluation model in the testing)
    count = 0

    loss_Fn=nn.CrossEntropyLoss()
    #nn.functional.cross_entropy()
    for batch_idx, (data, target) in enumerate(train_loader):
      #data is the image
      #target is the ground truth
        data, target = data.to(device), target.to(device)
        ############################
        #### Put your code here ####
        ############################
        Predict=model(data)
        loss=loss_Fn(Predict,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ###########################
        #### End of your codes ####
        ###########################
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test( model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    time0 = time.time()
    # Training settings
    batch_size = 128
    epochs = 25
    lr = 0.01

    no_cuda = False
    save_model = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(100)
    device = torch.device("cuda" if use_cuda else "cpu")

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    for epoch in range(1, epochs + 1):
        train( model, device, train_loader, optimizer, epoch)
        test( model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(),"cifar_lenet.pt")
    time1 = time.time()
    print ('Traning and Testing total excution time is: %s seconds ' % (time1-time0))
if __name__ == '__main__':
    main()
