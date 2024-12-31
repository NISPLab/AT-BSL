from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from models.ti_preact_resnet import ti_preact_resnet
from at_bsl_loss import pgd_loss
from pgd_attack import eval_adv_test_whitebox
from datasets.tiny_imagenet import load_tinyimagenet
parser = argparse.ArgumentParser(description='PyTorch Tiny-ImageNet AT-BSL Adversarial Training')
parser.add_argument('--aug', type=str, choices=['ra', 'none'], default='ra', metavar='N',
                    help='data augmentation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8./255.,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2./255.,
                    help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = 'model-tiny_imagenet-' + args.aug
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# setup data loader
train_loader, test_loader, samples_per_cls = load_tinyimagenet(data_dir='../data/tiny-imagenet',
                                              aug=args.aug,
                                              batch_size=args.batch_size,
                                              test_batch_size=args.test_batch_size)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = pgd_loss(model=model,
                           x_natural=data,
                           y=target,
                           samples_per_cls=samples_per_cls,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    model = ti_preact_resnet('preact-resnet18').to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    best_acc = 0
    best_rob = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        if epoch >= 75:
            acc, rob = eval_adv_test_whitebox(model, device, test_loader)
            if rob > best_rob:
                best_acc = acc
                best_rob = rob
                best_epoch = epoch
            print('best_acc:{:.4f} best_rob:{:.4f} best_epoch:{:d}'.format(best_acc, best_rob, best_epoch))
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'epoch{}.pt'.format(epoch)))


if __name__ == '__main__':
    main()
