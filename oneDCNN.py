import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import time

class bjkangNet(nn.Module):

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool3(self.conv3(x))
            x = self.conv4(x)
            x = self.conv5(x)
        return x.numel() 

    def __init__(self, input_channels, n_classes):
        super(bjkangNet,self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv1d(1, 32, 3 , stride = 1)
        
        self.conv2 = nn.Conv1d(32, 32, 3, stride = 1)
        self.conv3 = nn.Conv1d(32, 32, 3, stride = 1)

        self.pool3 = nn.MaxPool1d(kernel_size = 2, stride = 2)
        self.conv4 = nn.Conv1d(32, 32, 3, stride = 1)
        self.conv5 = nn.Conv1d(32, 32, 3, stride = 1)
        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.view(-1,self.features_size)
        x = self.fc1(x)
        X = self.fc2(x)
        return x

def train(model, train_loader, optimizer, epoch,iterator):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return iterator


def evaluate(model, test_loader,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            # pred = output.max(1, keepdim=True)[1]
            _, pred = torch.max(output,1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


if __name__=='__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    import load_mat_data

    EPOCHS = 400
    batch_size = 100
    test_sample = 0.3
    learning_rate = 0.001
    input_channels, n_classes, train_loader, test_loader = load_mat_data.load_mat('./Datasets/PaviaU/PaviaU.mat', './Datasets/PaviaU/PaviaU_gt.mat',batch_size, test_sample)
    model = bjkangNet(input_channels,n_classes).to(DEVICE)    
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    iterator = 0
    plt.ion()


    for epoch in range(1, EPOCHS + 1):
        iterator = train(model, train_loader, optimizer, epoch, iterator)
        test_loss, test_accuracy = evaluate(model, test_loader,epoch)
        if epoch == 90 : 
            break_point = 0
        
        print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
            epoch, test_loss, test_accuracy))


    
    PATH = './bjkangNet.pth'
    torch.save(model.state_dict(), PATH)