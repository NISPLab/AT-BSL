from datasets.datasets import IMBALANCECIFAR100, IMBALANCECIFAR10, \
    SMALLCIFAR10, SMALLCIFAR100
# from torchvision import transforms
import torchvision


def build_datasets(name='CIFAR10', mode='train', num_classes=10, imbalance_ratio=0.0, existing_ratio=1, root='../data', transform=None):
    
    if mode == 'train':
        train = True
    elif mode == 'test':
        train = False
    else:
        raise NameError('Invalid mode: {}'.format(mode))

    if imbalance_ratio > 0:
        if name == 'CIFAR10':
            dataset = IMBALANCECIFAR10(imbalance_ratio=imbalance_ratio,
                                        root=root, train=train, download=True,
                                        transform=transform)
        elif name == 'CIFAR100':
            dataset = IMBALANCECIFAR100(imbalance_ratio=imbalance_ratio,
                                         root=root, train=train, download=True,
                                         transform=transform)
        else:
            raise NameError('Invalid dataset name: {}'.format(name))
        
        samples_per_cls = dataset.img_num_list
    
    elif existing_ratio < 1:
        if name == 'CIFAR10':
            dataset = SMALLCIFAR10(existing_ratio=existing_ratio,
                                       root=root, train=train, download=True,
                                       transform=transform)
        else:
            dataset = SMALLCIFAR100(existing_ratio=existing_ratio,
                                   root=root, train=train, download=True,
                                   transform=transform)

        samples_per_cls = dataset.img_num_list
    else:
        if name == 'CIFAR10':
            dataset = torchvision.datasets.CIFAR10(root=root, train=train, 
                                                   download=True, 
                                                   transform=transform)
        elif name == 'CIFAR100':
            dataset = torchvision.datasets.CIFAR100(root=root, train=train, 
                                                    download=True, 
                                                    transform=transform)
        elif name == 'ImageNet':
            dataset = torchvision.datasets.ImageNet(root=root, train=train,
                                                     download=True,
                                                     transform=transform)
        else:
            raise NameError('Invalid dataset name: {}'.format(name))
        samples_per_cls = [5000] * num_classes

    return dataset, samples_per_cls