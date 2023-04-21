from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision import models
import torch
import torch.nn as nn
from trainer import train, test
import matplotlib.pyplot as plt
from model_wobn import resnet110

batch_size=128
lr_s = [0.0001]
epochs = 165 

train_dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size, num_workers=4, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for lr in lr_s: 
    model = resnet110().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_train_acc = []
    epoch_test_acc = []
    print(f"Expt: {lr}")
    for epoch in range(epochs):
        train_acc = train(model, train_loader, optimizer, criterion, epoch_train_acc, device)
        test_acc = test(model, test_loader, epoch_test_acc, device)
        print(f"Epoch {epoch}   Train Acc {train_acc}   Test Acc {test_acc}")
    # plt.imshow(train_img.cpu().numpy())
    # plt.savefig("target.png")
    file = open(f'lr={lr}_WoBN_Train.txt','w')
    file.write(str(epoch_train_acc))
    file.close()
    file = open(f'lr={lr}_WoBN_Test.txt','w')
    file.write(str(epoch_test_acc))
    file.close()
# plt.plot(epoch_train_acc, label='Accuracy Training')
# plt.legend()
# plt.show()
# plt.savefig("acc_train.png")

# plt.plot(epoch_train_acc, label='Accuracy Testing')
# plt.legend()
# plt.show()
# plt.savefig("acc_test.png")

    

