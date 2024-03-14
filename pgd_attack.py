from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

def _pgd_whitebox(model,
                  X,
                  y,
                  device,
                  epsilon=8. / 255.,
                  num_steps=20,
                  step_size=2. / 255.):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if True:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    out_pgd = model(X_pgd)
    err_pgd = (out_pgd.data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, device)
        robust_err_total += err_robust
        natural_err_total += err_natural

    print('Natual: ({:.2f}%)'.format(100. - 100. * natural_err_total / len(test_loader.dataset)))
    print('Robust: ({:.2f}%)'.format(100. - 100. * robust_err_total / len(test_loader.dataset)))
    accuracy = 1. - 1. * float(natural_err_total.cpu()) / len(test_loader.dataset)
    robustness = 1. - 1. * float(robust_err_total.cpu()) / len(test_loader.dataset)
    return accuracy, robustness

