import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
dirpath = os.getcwd()

BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 1
SAVE_PATH = dirpath + "/mnist_cnn.pt"


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

    print("Test average acccuracy {}/{}".format(corect, len(test_loader.dataset)))


def save(model):
    torch.save(model.state_dict(), SAVE_PATH)


def load_model(path):
    return torch.load(path)


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

    save(model)


if __name__ == "__main__":
    run_training()
