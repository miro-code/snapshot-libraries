#training structure inspired by https://github.com/timgaripov/dnn-mode-connectivity
import argparse
import os
import sys
import torch
from torch import nn
import time
import tabulate
import copy

import utils
import data
import models


def train_loop(dir, seed, dataset, data_path, model, batch_size, num_workers, transform, use_test, val_size, lr, momentum, wd, resume, epochs, save_freq, n_snapshots = 5):


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
    n_snapshots : int
        how many snapshots we return - epochs should be divisible by this
        
    Returns
    ----------

    model : List[torch.nn.module]
        Trained neural networks from the last n_return_models epochs. The final model is at the last index [-1]
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
    snapshot_duration = epochs // n_snapshots
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=snapshot_duration, eta_min=lr/100)
    
    columns = ['ep', 'mean_tr_loss', 'tr_acc', 'mean_te_loss', 'te_acc', 'time']
    test_res = {'mean_loss': None, 'accuracy': None}

    result = []

    for epoch in range(start_epoch, epochs + 1):
        time_ep = time.time()

        train_res = utils.train(loaders['train'], model, optimizer, loss_fn)
        lr_scheduler.step()
        test_res = utils.test(loaders['test'], model, loss_fn)
        
        if(epoch % snapshot_duration == 0):
            result.append(copy.deepcopy(model))
            utils.save_checkpoint(
                dir,
                epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                name = "snapshot"
            )
        
        elif(epoch % save_freq == 0 or epoch == epochs):
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
    return result
    


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
    parser.add_argument('--data_path', type=str, default="data", metavar='PATH',
                        help='path to datasets location (default: data)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default="ResNet20", metavar='MODEL',
                        help='model name (default: ResNet20)')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, default=25, metavar='N',
                        help='save frequency (default: 25)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

    args = parser.parse_args()

    config = vars(args)

    config["epochs"] = 300

    N_RUNS = 5
    
    model_dict = {i:train_loop(**config) for i in range(1,N_RUNS+1)}

    args.val_size = None
    args.use_test = True

    loaders, num_classes = data.loaders(
        dataset = args.dataset,
        path = args.data_path,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        transform_name = args.transform,
        use_test = args.use_test,
        val_size= args.val_size
    )
    train_loader, valid_loader = utils.split_dataloader(loaders["train"], 0.7)
    test_loader = loaders["test"]
    valid_models = {i: [utils.predictions(valid_loader, model)[0] for model in model_dict[i]] for i in model_dict.keys()}

    valid_targets = valid_loader.dataset.targets
    test_models = {i: [utils.predictions(test_loader, model)[0] for model in model_dict[i]] for i in model_dict.keys()}
    test_targets = test_loader.dataset.targets

    import ensemble_selection

    baseline_accuracy = ensemble_selection.accuracy([test_models[i][-1] for i in model_dict.keys()], test_targets)

    membership_flags = ensemble_selection.greedy_selection_without_replacement(N_RUNS, [m for i in model_dict.keys() for m in valid_models[i]], valid_targets, ensemble_selection.accuracy, minimize_metric_fn=False)
    print(membership_flags)
    import numpy as np
    selected_ensemble = np.array([m for i in model_dict.keys() for m in test_models[i]])[membership_flags]
    accuracy = ensemble_selection.accuracy(selected_ensemble, test_targets)

    print(f"baseline accuracy: {baseline_accuracy}, accuracy: {accuracy}")


    """   
 deprecated main method experiment
    #train_loop(**config)

    
    #for i in range(3, 6):
    #    config["dir"] = f"results/resnet56-3/{i}"
    #    config["resume"] = f"results/resnet56-3/{i}/checkpoint-50.pt"
    #    config["epochs"] = 70
    #    config["n_return_models"] = 20
    #    train_loop(**config)

    

    
    args.val_size = None
    args.use_test = True

    loaders, num_classes = data.loaders(
        dataset = args.dataset,
        path = args.data_path,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        transform_name = args.transform,
        use_test = args.use_test,
        val_size= args.val_size
    )

    model = models.ResNet20(num_classes = num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    CHECKPOINT_NAMES = [str(i) for i in range(51,71)]
    N_MODELS_PER_RUN = len(CHECKPOINT_NAMES)
    N_RUNS = 5
    model_dict = {i:[copy.deepcopy(model) for _ in range(N_MODELS_PER_RUN)] for i in range(1,N_RUNS+1)}

    for i,chkpt in enumerate(CHECKPOINT_NAMES):
        for m in model_dict.keys():
            checkpoint = torch.load(f"results/resnet56-3/{m}/checkpoint-{chkpt}.pt")
            model_dict[m][i].load_state_dict(checkpoint['model_state'])


    train_loader, valid_loader = utils.split_dataloader(loaders["train"], 0.7)
    test_loader = loaders["test"]
    valid_models = {i: [utils.predictions(valid_loader, model)[0] for model in model_dict[i]] for i in model_dict.keys()}

    valid_targets = valid_loader.dataset.targets
    test_models = {i: [utils.predictions(test_loader, model)[0] for model in model_dict[i]] for i in model_dict.keys()}
    test_targets = test_loader.dataset.targets

    import ensemble_selection

    baseline_accuracy = ensemble_selection.accuracy([test_models[i][-1] for i in model_dict.keys()], test_targets)

    membership_flags = ensemble_selection.greedy_selection_without_replacement(N_RUNS, [m for i in model_dict.keys() for m in valid_models[i]], valid_targets, ensemble_selection.accuracy, minimize_metric_fn=False)
    print(membership_flags)
    import numpy as np
    selected_ensemble = np.array([m for i in model_dict.keys() for m in test_models[i]])[membership_flags]
    accuracy = ensemble_selection.accuracy(selected_ensemble, test_targets)

    print(f"baseline accuracy: {baseline_accuracy}, accuracy: {accuracy}")
    """
    

    