import torch
import torch.nn.functional as F
from torch.autograd import Variable

def BSL(labels, logits, sample_per_class):
    spc = torch.tensor(sample_per_class).type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels)
    return loss

def pgd_loss(model,
                x_natural,
                y,
                samples_per_cls,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                distance='l_inf'):
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    loss = BSL(y, model(x_adv), samples_per_cls)
    return loss
