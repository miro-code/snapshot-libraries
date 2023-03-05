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


def predictions(test_loader, model):
    """
    Legacy code
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


def predict_and_store(file_path, test_loader, model, loss_fn):
    loss_fn = loss_fn(reduction = "none") #loss_fn returns loss for each sample
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = []
    losses = []
    targets = []

    output_per_layer = [[] for _ in model.children()] #each element is a list comparable to the aboves and corresponding to one layer

    for input_cpu, target_cpu in test_loader:
        input = input_cpu.to(device, non_blocking =True)
        target = target_cpu.to(device, non_blocking =True)
        output = model(input)
        layerwise_output = model.forward_per_layer(input)
        layerwise_output = [x.detach().cpu().numpy() for x in layerwise_output]
        for layer, layer_output in zip(output_per_layer, layerwise_output):
            layer.append(layer_output)
        loss = loss_fn(output, target)
        inputs.append(input_cpu)
        targets.append(target_cpu)
        losses.append(loss.detach().cpu().numpy())
    
    output_per_layer_dict = {}
    for i in range(len(output_per_layer)):
        output_per_layer_dict[f"layer_{i}"] = np.vstack(output_per_layer[i])

    inputs, losses, targets = np.vstack(inputs), np.concatenate(losses), np.concatenate(targets)
    np.savez(file_path, inputs = inputs, losses = losses, targets = targets, **output_per_layer_dict)

