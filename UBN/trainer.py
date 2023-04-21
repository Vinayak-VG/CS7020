import torch

def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred.data, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    return correct/total

def train(model, dataset, optimizer, loss, accu_train_epoch, device):
    
    train_loss_batch = []
    accu_train_batch = []
    model.train()
    for idx, (images, labels) in enumerate(dataset):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        train_loss = loss(output,labels)
        # train_loss_batch.append(train_loss)
        acc = accuracy(output, labels)
        accu_train_batch.append(acc)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    # epoch_train_losses.append(sum(train_loss_batch)/len(dataset))
    accu_train_epoch.append(sum(accu_train_batch)/len(dataset))
    # print(f"Train Epoch Loss: {sum(train_loss_batch)/len(dataset)}   Train Epoch Accuracy: {sum(accu_train_batch)/len(dataset)}")
    return sum(accu_train_batch)/len(dataset)

def test(model, dataset, accu_test_epoch, device):
    
    accu_test_batch = []
    model.eval()
    for idx, (images, labels) in enumerate(dataset):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            # test_loss = criterion(output,labels)
            # test_loss_batch.append(test_loss)
            acct = accuracy(output, labels)
            accu_test_batch.append(acct)
    # epoch_test_losses.append((sum(test_loss_batch))/63)
    accu_test_epoch.append((sum(accu_test_batch))/len(dataset))
    # print(f"Test Epoch Accuracy: {(sum(accu_test_batch))/63}")
    return sum(accu_test_batch)/len(dataset)
    
