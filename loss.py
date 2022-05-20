import sys
import torch

import ChamferDistancePytorch.chamfer2D.dist_chamfer_2D as dist_chamfer_2D
import ChamferDistancePytorch.chamfer6D.dist_chamfer_6D as dist_chamfer_6D
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D

sys.path.insert(1, './ChamferDistancePytorch')
ch6d = dist_chamfer_6D.chamfer_6DDist()
ch2d = dist_chamfer_2D.chamfer_2DDist()
ch3d = dist_chamfer_3D.chamfer_3DDist()


def chamfer(a, b):

    if a.size(-1) == 6:
        dist1, dist2, idx1, idx2 = ch6d(a.contiguous(), b.contiguous())

    if a.size(-1) == 2:
        dist1, dist2, idx1, idx2 = ch2d(a.contiguous(), b.contiguous())

    if a.size(-1) == 3:
        dist1, dist2, idx1, idx2 = ch3d(a.contiguous(), b.contiguous())

    return torch.mean(torch.sqrt(dist1) + torch.sqrt(dist2))


def cycle(x2D, x2D_cycle, y6D, y6D_cycle):

    loss = torch.nn.MSELoss()

    return loss(x2D, x2D_cycle) + loss(y6D, y6D_cycle)


def repulse(pdf, sigma):

    pdf = pdf[0]
    n = pdf.size(0)
    d = pdf.size(1)

    x = pdf.unsqueeze(0).expand(n, n, d)
    y = pdf.unsqueeze(1).expand(n, n, d)

    dist = torch.norm(x - y, dim=2)
    dist = torch.exp(-dist / (2 * sigma))
    dist = (torch.sum(dist) - n) / (n * n)
    return dist
