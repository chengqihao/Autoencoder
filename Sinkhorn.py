import warnings
import numpy as np
import torch


def construct_1D_M(image_size, device):
    tempM = torch.arange(image_size * image_size, dtype=torch.float64, device=device)
    tempM2 = tempM - tempM[:, np.newaxis]
    tempM3 = torch.triu(tempM2)
    pM = tempM3 + tempM3.t()
    pM = pM / (5 * pM.max())
    return pM


def construct_2D_M(image_size, device):
    tempM = torch.zeros(image_size * image_size, 2, dtype=torch.int8, device=device)
    tempM[:, 1] = torch.arange(image_size * image_size) % image_size
    tempM[:, 0] = torch.arange(image_size * image_size) / image_size
    tempM = tempM.float()
    tempM2 = tempM
    MM1 = torch.cdist(tempM, tempM2, p=1)
    MM = MM1.view(-1, image_size * image_size, image_size * image_size)
    MM /= (5 * MM.max())
    return MM


def normalizations(original, target):
    bs, k = original.size()
    sumo = original.sum(dim=1)
    sumo = sumo.view(bs, -1)
    sumo = sumo.repeat(1, k)

    sumt = target.sum(dim=1)
    sumt = sumt.view(bs, -1)
    sumt = sumt.repeat(1, k)

    tr_original = original / sumo
    tr_target = target / sumt
    tr_original = tr_original + 1e-7
    tr_target = tr_target + 1e-7

    sumo = tr_original.sum(dim=1)
    sumo = sumo.view(bs, -1)
    sumo = sumo.repeat(1, k)

    sumt = tr_target.sum(dim=1)
    sumt = sumt.view(bs, -1)
    sumt = sumt.repeat(1, k)
    tr_original = tr_original / sumo
    tr_target = tr_target / sumt
    return tr_original, tr_target


def Sinkhorn(original, target, M, maxiter, eps, device):
    bs, k = original.size()
    tr_original, tr_target = normalizations(original, target)
    tr_original = tr_original.view(-1, k, 1)
    tr_target = tr_target.view(-1, 1, k)
    psi = (torch.ones([bs, 1, k], device=device) / k)
    phi = torch.zeros([bs, k, 1], device=device)
    G = torch.exp(-M / eps)
    for i in range(maxiter):
        phi = tr_original / (G * psi).sum(-1, keepdim=True)
        psi = tr_target / (G * phi).sum(-2, keepdim=True)
        if torch.any(G * psi == 0) or torch.any(G * phi == 0) or torch.any(torch.isnan(psi)) or torch.any(
                torch.isnan(phi)) or torch.any(
                torch.isinf(psi)) or torch.any(torch.isinf(phi)):
            warnings.warn('Warning: numerical errors at iteration %d' % i)
            break
    phi = phi.view(-1, 28 * 28)
    psi = psi.view(-1, 28 * 28)
    alpha = -(torch.log(phi) + 0.5) * eps
    beta = -(torch.log(psi) + 0.5) * eps
    return alpha, beta
