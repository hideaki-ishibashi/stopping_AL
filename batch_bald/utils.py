import numpy as np
import os

import torch
import torchvision


def sample_normal(mean, variance, num_samples=1, squeeze=False):
    """
    Reparameterized sample from a multivariate Normal distribution
    :param mean: (torch.tensor) Mean of the distribution
    :param variance: (torch.tensor) Variance of the distribution
    :param num_samples: (int) Number of samples to take
    :param squeeze: (bool) Squeeze unnecessary dimensions
    :return: (torch.tensor) Samples from Gaussian distribution
    """
    noise = to_gpu(torch.nn.init.normal_(torch.FloatTensor(num_samples, *mean.shape)))
    samples = torch.sqrt(variance + 1e-6) * noise + mean

    if squeeze and num_samples == 1:
        samples = torch.squeeze(samples, dim=0)
    return samples


def gaussian_kl_diag(mean1, variance1, mean2, variance2):
    """
    KL-divergence between two diagonal Gaussian distributions
    :param mean1: (torch.tensor) Mean of first distribution
    :param variance1: (torch.tensor) Variance of first distribution
    :param mean2: (torch.tensor) Mean of first distribution
    :param variance2: (torch.tensor) Variance of second distribution
    :return: (torch.tensor) Value of KL-divergence
    """
    return -0.5 * torch.sum(1 + torch.log(variance1) - torch.log(variance2) - variance1 / variance2
                            - ((mean1 - mean2)**2) / variance2, dim=-1)


def smart_gaussian_kl(mean1, covariance1, mean2, covariance2):
    """
    Compute the KL-divergence between two Gaussians
    :param mean1: mean of q
    :param covariance1: covariance of q
    :param mean2: mean of q
    :param covariance2: covariance of p, diagonal
    :return: kl term
    """

    k = mean1.numel()
    assert mean1.shape == mean2.shape
    assert covariance1.shape[0] == covariance1.shape[1] == k
    assert covariance2.shape[0] == covariance2.shape[1] == k
    mean1, mean2 = mean1.view(-1, 1), mean2.view(-1, 1)
    slogdet_diag = lambda a: torch.sum(torch.log(torch.diag(a)))

    variance1 = torch.diag(covariance1)
    variance2 = torch.diag(covariance2)
    if torch.equal(torch.diag(variance1), covariance1):
        return gaussian_kl_diag(mean1.flatten(), variance1, mean2.flatten(), variance2).squeeze()

    covariance2_inv = torch.diag(1. / torch.diag(covariance2))
    x = mean2 - mean1
    kl = 0.5 * (torch.trace(covariance2_inv @ covariance1) +
                x.t() @ covariance2_inv @ x -
                k + slogdet_diag(covariance2) - torch.slogdet(covariance1)[1])
    return kl.squeeze()


def sample_lr_gaussian(mean, F, variance, num_samples, squeeze=False):
    """
    Generate reparameterized samples from a full Gaussian with a covariance of
    FF' + diag(variance)
    :param mean: (tensor) mean of the distribution
    :param F: (tensor) low rank parameterization of correlation structure
    :param variance: (tensor) variance, i.e., diagonal of the covariance matrix
    :param num_samples: (int) number of samples to take from the distribution
    :param squeeze: (bool) squeeze the samples if only one
    :return: sample from the distribution
    """

    epsilon_f = to_gpu(torch.nn.init.normal_(torch.FloatTensor(1, F.shape[1], num_samples)))
    epsilon_v = to_gpu(torch.nn.init.normal_(torch.FloatTensor(1, variance.shape[0], num_samples)))

    m_h_tiled = mean[:, :, None].repeat(1, 1, num_samples)  # N x k x L
    Fz = F @ epsilon_f
    Vz = torch.diag(variance / 2.) @ epsilon_v
    local_reparam_samples = Fz + Vz + m_h_tiled
    samples = local_reparam_samples.permute([2, 0, 1])  # L x N x k

    if squeeze and num_samples == 1:
        samples = torch.squeeze(samples, dim=0)

    return samples


def to_gpu(*args):
    return [arg for arg in args] if len(args) > 1 else args[0]
    # return [arg.cuda() for arg in args] if len(args) > 1 else args[0].cuda()
