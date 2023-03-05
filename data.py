#based on https://github.com/timgaripov/dnn-mode-connectivity
import os
import torch
import torchvision
import torchvision.transforms as transforms


class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])
        class Normalize:
            train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])


    CIFAR100 = CIFAR10

    class MNIST:
        class NoTransform:
            train = transforms.ToTensor()
            test = transforms.ToTensor()
        class Normalize:
            train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081]),
            ])
    class FashionMNIST:
        class NoTransform:
            train = transforms.ToTensor()
            test = transforms.ToTensor()
        class Normalize:
            train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2860], std=[0.3530]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2860], std=[0.3530]),
            ])


def loaders(dataset, path, batch_size, num_workers, transform_name="Normalize", val_size = 0.1, use_test=False,
            shuffle_train=True):
    """ return train and test loader for the given dataset; if use_test is False then val_size determines how many of the train samples are in the test_loader
    dataset : str
        name of the dataset (CIFAR or CIFAR100)
    path : str
        folder where the dataset is stored
    batch_size : int 
    num_workers : int
    transform_name : str
        name of the transform to be applied to the data (e.g. Normalize - see data.py)
    val_size : float
        only use if use_test is false; defines the ratio of samples used in the test_loader
    use_test : bool
        whether to use the test set that is being held out during development

    
    """

    if(use_test and val_size is not None):
        raise ValueError("Can't use test and pass val_size simultaneously")
    if(val_size == 0):
        raise ValueError("val_size can't be 0 - set use_test=True to obtain full training set")
    ds = getattr(torchvision.datasets, dataset)
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)
    num_classes = max(train_set.targets) + 1
    if(type(num_classes) == torch.Tensor):
        num_classes = num_classes.item()
    if use_test:
        print('You are about to run models on the test set. Please ensure that this is intended')
        test_set = ds(path, train=False, download=True, transform=transform.test)
    else:
        n_val = int(len(train_set) * val_size)
        print(f"Using train {len(train_set) - n_val} + validation {n_val}")
        train_set.data = train_set.data[:-n_val]
        train_set.targets = train_set.targets[:-n_val]

        test_set = ds(path, train=True, download=True, transform=transform.test)
        test_set.train = False
        test_set.data = test_set.data[-n_val:]
        test_set.targets = test_set.targets[-n_val:]

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
        
    return {
               'train': train_loader,
               'test': test_loader,
           }, num_classes

if(__name__ == "__main__"):
    loaders = loaders("FashionMNIST", "data", 128, 4, "Normalize", 0.3, False)