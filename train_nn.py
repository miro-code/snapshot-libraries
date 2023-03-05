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


def main(args):
    if(args.use_test and args.val_size != 1.0):
        raise ValueError("Can't scale validation set when using test")
    
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, 'command.sh'), 'a') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    loaders, num_classes = data.loaders(
        dataset = args.dataset,
        path = args.data_path,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        transform_name = args.transform,
        use_test = args.use_test,
        val_size=args.val_size
    )

 
    
    model_class = getattr(models, args.model)
    model = model_class(num_classes = num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
    )
    loss_fn = nn.CrossEntropyLoss

    start_epoch = 1
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        utils.save_checkpoint(
        args.dir,
        start_epoch - 1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
        )
    columns = ['ep', 'mean_tr_loss', 'tr_acc', 'mean_te_loss', 'te_acc', 'time']
    test_res = {'mean_loss': None, 'accuracy': None}
    for epoch in range(start_epoch, args.epochs + 1):
        time_ep = time.time()

        train_res = utils.train(loaders['train'], model, optimizer, loss_fn)
        test_res = utils.test(loaders['test'], model, loss_fn)

        if(epoch % args.save_freq == 0 or epoch == args.epochs):
            utils.save_checkpoint(
                args.dir,
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
    
    utils.predict_and_store(args.dir + "/validation.npz", loaders['test'], model, loss_fn)


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
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

    args = parser.parse_args()


    main(args)
        