# Usage: python brew_poison.py --optimization private --gradient_clip 1.0 --gradient_noise 0.01 --noise_type laplace --log_gradients

import torch


def add_laplace_noise(model, clip, noise_scale):
    """Replace Gaussian gradient noise with Laplace noise scaled by clip * noise_scale."""
    dist = torch.distributions.Laplace(torch.tensor(0.0), torch.tensor(clip * noise_scale))
    for param in model.parameters():
        if param.grad is not None:
            noise_sample = dist.sample(param.grad.shape).to(param.grad.device)
            param.grad += noise_sample


def log_gradient_stats(model, step, epoch, writer=None):
    """Return a dict of per-parameter gradient L2 norms at the current moment."""
    stats = {'step': step, 'epoch': epoch}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.detach().pow(2).sum().sqrt().item()
            stats[f'{name}_l2'] = norm
    return stats
