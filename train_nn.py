#inspired by https://github.com/timgaripov/dnn-mode-connectivity
import argparse
import os
import sys
import torch
from torch import nn
import time
import tabulate

import utils
import data
import models

def train_loop(dir, seed, dataset, data_path, model, batch_size, num_workers, transform, use_test, val_size, lr, momentum, wd, resume, epochs, save_freq):


    """ Trains a neural network for multiple epochs

    Parameters
    ----------

    dir : str
        folder where checkpoints are stored
    seed : int
    dataset : str
        name of the dataset we train on (CIFAR10 or CIFAR100)
    model : str
        name of the model to use (see models/)
    data_path : str
        folder where we store the dataset
    batch_size : int
    num_workers : int
    transform : str
        name of the transform applied to the dataset after loading it (defined in data.py) examples are Normalize and VGG
    use_test : bool
        whether to use the test set
    val_size : int
        FILL IN LATER
    lr : float
    momentum : float
    wd : float
    resume : str
        if not None the training is resumed from the checkpoint in "{dir}/{resume}
    epochs : int
        how many epochs the network should be trained for in total (f.e. if epochs = 10 and we resume training after 7 epochs the method will run for 3 more epochs)
    save_freq : int
        how frequently the model is stored in checkpoints (in epochs)

    Returns
    ----------

    model : torch.nn.module
        trained neural network
    """

    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, 'command.sh'), 'a') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    loaders, num_classes = data.loaders(
        dataset = dataset,
        path = data_path,
        batch_size = batch_size,
        num_workers = num_workers,
        transform_name = transform,
        use_test = use_test,
        val_size= val_size
    )

    model_class = getattr(models, model)
    model = model_class(num_classes = num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=wd
    )
    loss_fn = nn.CrossEntropyLoss

    start_epoch = 1
    if resume is not None:
        print('Resume training from %s' % resume)
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        utils.save_checkpoint(
        dir,
        start_epoch - 1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
        )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=start_epoch - 2)
    
    columns = ['ep', 'mean_tr_loss', 'tr_acc', 'mean_te_loss', 'te_acc', 'time']
    test_res = {'mean_loss': None, 'accuracy': None}
    for epoch in range(start_epoch, epochs + 1):
        time_ep = time.time()

        train_res = utils.train(loaders['train'], model, optimizer, loss_fn)
        lr_scheduler.step()
        test_res = utils.test(loaders['test'], model, loss_fn)

        if(epoch % save_freq == 0 or epoch == epochs):
            utils.save_checkpoint(
                dir,
                epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        time_ep = time.time() - time_ep

        values = [epoch, train_res['mean_loss'], train_res['accuracy'], test_res['mean_loss'],
                  test_res['accuracy'], time_ep]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 1 or epoch == start_epoch:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
    
    if epochs % save_freq != 0:
        utils.save_checkpoint(
            dir,
            epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )
    return model
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNN curve training')
    parser.add_argument('--dir', type=str, default='/tmp/curve/',
                        help='training directory (default: /tmp/curve/)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
    parser.add_argument('--val_size', type=float, default=0.1, help='ration of the validation set (default: 0.1)')

    parser.add_argument('--transform', type=str, default='Normalize', metavar='TRANSFORM',
                        help='transform name (default: Normalize)')
    parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                        help='model name (default: None)')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                        help='save frequency (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

    args = parser.parse_args()

    config = vars(args)
    train_loop(**config)
        