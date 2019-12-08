import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from .util import batch_multiply, batch_clamp
import os
dirpath = os.getcwd()

BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 1
STEP_SIZE = 0.3
CLIP_MIN = 0
CLIP_MAX = 1
SAVE_PATH = dirpath + "/mnist_cnn.pt"


def fgsm(model, data, target, epsilon):
    pert = torch.zeros_like(data, requires_grad=True)
    output = model(data+pert)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    return epsilon * pert.grad.detach().sign()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input):
        output = F.relu(self.conv1(input))
        output = self.conv2(output)
        output = F.max_pool2d(output, 2)
        output = torch.flatten(output, 1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output


def get_data():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # get loss
        output = model(data)
        loss = criterion(output, target)
        # backpropagate
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    corect = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            corect += pred.eq(target.view_as(pred)).sum().item()

    print("Test acccuracy {}/{}".format(corect, len(test_loader.dataset)))


def test_fgsm(model, device, test_loader):
    model.eval()
    corect = 0
    for data, target in test_loader:
        adv = fgsm(model, data, target, STEP_SIZE)
        output = model(adv)
        pred = output.argmax(dim=1, keepdim=True)
        corect += pred.eq(target.view_as(pred)).sum().item()
    print("Test FGSM accuracy {}/{}".format(corect, len(test_loader
                                                        .dataset)))


def train_fgsm(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        datax = fgsm(model, data, target, STEP_SIZE)
        data, target = datax.to(device), target.to(device)
        optimizer.zero_grad()
        # get loss
        output = model(data)
        loss = criterion(output, target)
        # backpropagate
        loss.backward()
        optimizer.step()


def save(model):
    torch.save(model.state_dict(), SAVE_PATH)


def load_model(path, device):
    model = Net()
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


def run_training():
    train_loader, test_loader = get_data()
    device = torch.device("cpu")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for i in range(EPOCHS):
        # train
        train_epoch(model, device, train_loader, optimizer, criterion)
        # test
        test(model, test_loader)


def run_adv_training():
    train_loader, test_loader = get_data()
    device = torch.device("cpu")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for i in range(EPOCHS):
        train_fgsm(model, device, train_loader, optimizer, criterion)
        test_fgsm(model, device, test_loader)
        test(model, test_loader)


if __name__ == "__main__":
    # device = torch.device("cpu")
    # model = load_model(SAVE_PATH, device)
    # _, test_loader = get_data()
    # test_fgsm(model, device, test_loader)

    run_adv_training()
