import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F



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


