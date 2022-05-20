import os
import torch
import numpy as np

from dataloader import SingleShape, RandomSampler
from model import Model
from loss import chamfer, cycle, repulse
from gradient import compute_jac, normals_from_jacobian, isometric

from utils import create_log, parse_args, save_params


# ------------------------------------------------------------------------------
# parse argument
# ------------------------------------------------------------------------------
settings = parse_args()
for key in settings.__dict__:
    print(f'{key: <20} : {settings.__dict__[key]}')
create_log(settings)


# ------------------------------------------------------------------------------
# dataloader
# ------------------------------------------------------------------------------
dataset = SingleShape(settings)
data = dataset[0].float().unsqueeze(0).cuda()
sampler = torch.utils.data.DataLoader(RandomSampler(settings, len(data[0])),
                                      batch_size=1,
                                      num_workers=4)

# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
model = Model(settings).cuda()

# ------------------------------------------------------------------------------
# latent vect
# ------------------------------------------------------------------------------
# mean = 0.0
# std  = 1 / np.sqrt(settings.nlatent)
# lat_vecs_2D = torch.nn.Embedding(1, settings.nlatent, max_norm=1)
# lat_vecs_6D = torch.nn.Embedding(1, settings.nlatent, max_norm=1)
# torch.nn.init.normal_(lat_vecs_2D.weight.data, mean, std)
# torch.nn.init.normal_(lat_vecs_6D.weight.data, mean, std)


# ------------------------------------------------------------------------------
# optimizer
# ------------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ------------------------------------------------------------------------------
# index
# ------------------------------------------------------------------------------
SELECT_X = torch.zeros(1, settings.npoint, 3).cuda()
SELECT_Y = torch.zeros(1, settings.npoint, 3).cuda()
SELECT_Z = torch.zeros(1, settings.npoint, 3).cuda()
SELECT_X[:, :, 0] = 1
SELECT_Y[:, :, 1] = 1
SELECT_Z[:, :, 2] = 1
INDEX = [SELECT_X, SELECT_Y, SELECT_Z]

# ------------------------------------------------------------------------------
# training logs
# ------------------------------------------------------------------------------
training_loss = []
for ep, idx in enumerate(sampler):

    optimizer.zero_grad()

    # --------------------------------------------------------------------------
    # training data + randomisation during training at every 1000 epoch
    # --------------------------------------------------------------------------
    y6D = data[:, idx[0], :]
    if ep % 1000 == 0:
        data = data[torch.randperm(len(data))]

    # --------------------------------------------------------------------------
    # parameterization path
    # --------------------------------------------------------------------------
    x2D = model.pdf
    x3D = model.parameterization(input=x2D, code=None)
    jac = compute_jac(x2D, x3D, INDEX)
    x3D_n = normals_from_jacobian(jac)
    x3D_n *= settings.normal_factor
    x6D = torch.cat((x3D, x3D_n), -1).contiguous()
    x2D_cycle = model.chart(input=x6D, code=None)

    # --------------------------------------------------------------------------
    # chart model in 6D
    # --------------------------------------------------------------------------
    y2D = model.chart(input=y6D, code=None)
    y3D_cycle = model.parameterization(input=y2D, code=None)
    y3D_cycle_n = normals_from_jacobian(compute_jac(y2D, y3D_cycle, INDEX))
    y3D_cycle_n *= settings.normal_factor
    y6D_cycle = torch.cat((y3D_cycle, y3D_cycle_n), -1)

    # --------------------------------------------------------------------------
    # Computing losses
    # --------------------------------------------------------------------------
    loss = chamfer(x6D, y6D)
    loss += chamfer(x2D, y2D)
    loss += cycle(x2D, x2D_cycle, y6D, y6D_cycle) * settings.cycle_factor
    sigma_rep = settings.repulsive_factor / np.sqrt(settings.npoint)
    loss += repulse(x2D, sigma_rep) * settings.repulsive_factor
    loss += isometric(jac[0], jac[1], jac[2]) * settings.regul_factor
    training_loss.append(loss.item())

    # --------------------------------------------------------------------------
    # backward and optimizer step
    # --------------------------------------------------------------------------
    loss.backward()
    optimizer.step()

    # --------------------------------------------------------------------------
    # display loss
    # --------------------------------------------------------------------------
    # if ep % 250 == 0 and ep != 0:
    display = f'Training ID : {settings.name} - '
    display += f'ep {ep:5d} / {settings.nepoch} - '
    display += f'L = {training_loss[-1]:.2e}'
    print(display)

    if ep == int(settings.nepoch * .9):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] /= 10

# ------------------------------------------------------------------------------
# training outputs
# ------------------------------------------------------------------------------
loss = {'training loss': training_loss,
        'chamfer 6d': reconstruction_loss, }

path = os.path.join('Log', settings.exp, 'time.txt')
with open(path, 'a') as f:
    f.write(f'{settings.name} - {end_time}\n')

print('Saving results and parameters')
modules = {'model': model}

# ------------------------------------------------------------------------------
# save params
# ------------------------------------------------------------------------------
save_params(settings, modules, loss)
