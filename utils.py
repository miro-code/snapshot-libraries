import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from copy import deepcopy


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)



def train(train_loader, model, optimizer, loss_fn):
    loss_fn = loss_fn() #call loss function class

    loss_sum = 0.0
    correct = 0.0
    
    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for input, target in train_loader:
        input = input.to(device, non_blocking =True)
        target = target.to(device, non_blocking =True)

        output = model(input)
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'mean_loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct / len(train_loader.dataset),
    }




def test(test_loader, model, loss_fn):
    loss_fn = loss_fn() #call loss function class

    loss_sum = 0.0
    correct = 0.0

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for input, target in test_loader:
        input = input.to(device, non_blocking =True)
        target = target.to(device, non_blocking =True)

        output = model(input)
        loss = loss_fn(output, target)

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'mean_loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct / len(test_loader.dataset),
    }


def split_dataloader(loader, split_ratio):
    """takes a dataloader and returns two with the data split amongst them

    loader : torch.utils.data.DataLoader
    split_ratio : float
        ratio of data to allocate to the first loader returned

    Returns
    --------
    (torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """

    batch_size = loader.batch_size
    num_workers = loader.num_workers
    shuffle = False #cant check this
    pin_memory = loader.pin_memory

    dataset1 = deepcopy(loader.dataset)
    dataset2 = deepcopy(loader.dataset)

    split_index = int(len(dataset1) * split_ratio)

    dataset1.data = dataset1.data[:split_index]
    dataset1.targets = dataset1.targets[:split_index]

    dataset2.data = dataset2.data[split_index:]
    dataset2.targets = dataset2.targets[split_index:]
    

    loader1 = torch.utils.data.DataLoader(
        dataset1,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    loader2 = torch.utils.data.DataLoader(
        dataset2,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return loader1, loader2


def predictions(test_loader, model):
    """
    get ndarrays of model predictions on dataloader

    test_loader : torch.utils.data.DataLoader  
    model : torch.nn.module

    Returns
    -------

    (ndarray, ndarray) - probability predictions, true labels
    """
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda(non_blocking =True)
        output = model(input)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)