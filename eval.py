import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
from torchvision.transforms import v2
from torch.autograd import Variable
import torch.optim as optim
from models.resnet import *
from models.wideresnet import *
def cwloss(output, target,confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def _pgd_whitebox(model,
                  X,
                  y,
                  device,
                  alpha,
                  attack_method,
                  ):
    if attack_method == 'None':
        out = model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        return err
    if attack_method == 'FGSM':
        epsilon=8. / 255.
        num_steps=1
        step_size=8. / 255.
    elif attack_method == 'PGD' or attack_method == 'CW':
        epsilon=8. / 255.
        num_steps=20
        step_size=2. / 255.
    
    X_pgd = Variable(X.data, requires_grad=True)
    if True:
        if attack_method == 'FGSM':
            random_noise = 0.001 * torch.randn(*X_pgd.shape).cuda().detach()
        else:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            if attack_method == 'CW':
                loss = cwloss(model(X_pgd), y, num_classes=num_classes)
            else:
                loss = nn.CrossEntropyLoss()(alpha * model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    out_pgd = model(X_pgd)
    err_pgd = (out_pgd.data.max(1)[1] != y.data).float().sum()
    return err_pgd

def eval_adv_test_whitebox(model, device, test_loader, attack_method, alpha=1.):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _pgd_whitebox(model, X, y, device, alpha, attack_method)
        robust_err_total += err_robust

    robustness = 1 - robust_err_total / len(test_loader.dataset)
    print('Robust: ({:.2f}%)'.format(100 * robustness))
    return robustness

def evaluate_attack(model, test_loader, atk, atk_name):
    test_acc = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to('cuda'), y.to('cuda')

        if atk_name == 'clean':
            X_adv = X
        else:
            X_adv = atk(X, y)  # advtorch

        with torch.no_grad():
            output = model(X_adv)
        test_acc += (output.max(1)[1] == y).sum().item()

    robustness = test_acc / len(test_loader.dataset)
    print('Robust: ({:.2f}%)'.format(100 * robustness))
    return robustness

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--arch', type=str, default='res')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--whitebox', type=bool, default=True)
    parser.add_argument('--lsa', type=bool, default=True)
    parser.add_argument('--aa', type=bool, default=True)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    transform_list = [v2.ToTensor()]
    transform_chain = v2.Compose(transform_list)

    model_path = args.model_path
    if args.dataset == 'cifar10':
        item = datasets.CIFAR10(root='../data', train=False, transform=transform_chain, download=True)
        test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'cifar100':
        item = datasets.CIFAR100(root='../data', train=False, transform=transform_chain, download=True)
        test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_classes = len(test_loader.dataset.classes)
    # load model
    if args.arch == 'res':
        model = ResNet18(num_classes=num_classes)
    elif args.arch == 'wrn':
        model = WideResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    import time
    # white-box attack
    robustness_list = []
    if args.whitebox:
        attack_method_list = ['None', 'FGSM', 'PGD', 'CW']
        for i, attack_method in enumerate(attack_method_list):
            print(attack_method)
            start_time = time.time()
            robustness = eval_adv_test_whitebox(model, device, test_loader, attack_method)
            end_time = time.time()
            print('{}: {:.1f} seconds'.format(attack_method, end_time - start_time))
            robustness_list.append(robustness)

    if args.lsa:
        attack_method_list = ['PGD']
        for i, attack_method in enumerate(attack_method_list):
            print('lsa', attack_method)
            start_time = time.time()
            robustness = eval_adv_test_whitebox(model, device, test_loader, attack_method, alpha=10.)
            end_time = time.time()
            print('{}: {:.1f} seconds'.format(attack_method, end_time - start_time))
            robustness_list.append(robustness)

    # load attack
    if args.aa:
        import torchattacks
        print('autoattack')
        start_time = time.time()
        atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=num_classes)
        robustness = evaluate_attack(model, test_loader, atk, 'autoattack')
        end_time = time.time()
        print('{}: {:.1f} seconds'.format('autoattack', end_time - start_time))
        robustness_list.append(robustness)
    print(model_path)
    for i in range(len(robustness_list)):
        print('{:.4f}\t'.format(robustness_list[i]), end='')
    print()

                
