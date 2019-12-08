import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from util import batch_multiply, batch_clamp
import os
dirpath = os.getcwd()

BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 1
STEP_SIZE = 0.3
CLIP_MIN = 0
CLIP_MAX = 1
SAVE_PATH = dirpath + "/mnist_cnn.pt"
NB_ITER = 2
EPS_ITER = 0.01


def pgd(nb_iter, data, target, eps, eps_iter, model, delta_init=None):
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(data)

    delta.requires_grad_()
    for _ in range(nb_iter):
        output = model(data+delta)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        grad_sign = delta.grad.data.sign()
        delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
        delta.data = batch_clamp(eps, delta.data)
        delta.data = torch.clamp(data.data + delta.data, 0, 1
                                 ) - data.data
        delta.grad.data.zero_()
    data = torch.clamp(data + delta, 0, 1)
    return data


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


def test(model, test_loader):
    model.eval()
    corect = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            corect += pred.eq(target.view_as(pred)).sum().item()

    print("Test acccuracy {}/{}".format(corect, len(test_loader.dataset)))


def test_pgd(model, device, test_loader):
    model.eval()
    corect = 0
    for data, target in test_loader:
        adv = pgd(NB_ITER, data, target, STEP_SIZE,
                  EPS_ITER, model)
        output = model(adv)
        pred = output.argmax(dim=1, keepdim=True)
        corect += pred.eq(target.view_as(pred)).sum().item()
    print("Test PGD accuracy {}/{}".format(corect, len(test_loader
                                                       .dataset)))


def train_pgd(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        datax = pgd(NB_ITER, data, target, STEP_SIZE,
                    EPS_ITER, model)
        data, target = datax.to(device), target.to(device)
        optimizer.zero_grad()
        # get loss
        output = model(data)
        loss = criterion(output, target)
        # backpropagate
        loss.backwa
        rd()
        optimizer.step()


def save(model):
    torch.save(model.state_dict(), SAVE_PATH)


def load_model(path, device):
    model = Net()
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


def run_adv_training():
    train_loader, test_loader = get_data()
    device = torch.device("cpu")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for i in range(EPOCHS):
        train_pgd(model, device, train_loader, optimizer, criterion)
        test_pgd(model, device, test_loader)
        test(model, test_loader)


if __name__ == "__main__":
    # device = torch.device("cpu")
    # model = load_model(SAVE_PATH, device)
    # _, test_loader = get_data()
    # test_fgsm(model, device, test_loader)

    run_adv_training()
