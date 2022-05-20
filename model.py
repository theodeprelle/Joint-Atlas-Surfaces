import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Map2Dto3D(nn.Module):
    def __init__(self, nlatent=256):

        super(Map2Dto3D, self).__init__()
        self.nlatent = nlatent
        self.conv1 = torch.nn.Conv1d(2, self.nlatent, 1)
        self.conv2 = torch.nn.Conv1d(self.nlatent, self.nlatent, 1)
        self.conv3 = torch.nn.Conv1d(self.nlatent, self.nlatent // 2, 1)
        self.conv4 = torch.nn.Conv1d(self.nlatent // 2, self.nlatent // 4, 1)
        self.conv5 = torch.nn.Conv1d(self.nlatent // 4, 3, 1)

    def forward(self, x, latent=None):

        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        if latent is not None:
            x += latent.unsqueeze(2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x.transpose(2, 1)


class Map6Dto2D(nn.Module):

    def __init__(self, nlatent=256):

        super(Map6Dto2D, self).__init__()
        self.nlatent = nlatent
        self.conv1 = torch.nn.Conv1d(6, self.nlatent, 1)
        self.conv2 = torch.nn.Conv1d(self.nlatent, self.nlatent, 1)
        self.conv3 = torch.nn.Conv1d(self.nlatent, self.nlatent // 2, 1)
        self.conv4 = torch.nn.Conv1d(self.nlatent // 2, self.nlatent // 4, 1)
        self.conv5 = torch.nn.Conv1d(self.nlatent // 4, 2, 1)

    def forward(self, x, latent=None):
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        if latent is not None:
            x += latent.unsqueeze(2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x.transpose(2, 1)


class Model(nn.Module):

    def __init__(self, settings):

        super(Model, self).__init__()

        self.npoint = settings.npoint
        self.nlatent = settings.nlatent
        self.std = settings.sigma_factor / np.sqrt(self.npoint)
        self.patchDim = 2

        # encoder and decoder modules
        # ----------------------------------------------------------------------
        self.map2Dto3D = Map2Dto3D(self.nlatent)
        self.map6Dto2D = Map6Dto2D(self.nlatent)
        # ----------------------------------------------------------------------

        # pdf
        # ----------------------------------------------------------------------
        self.pdf = torch.nn.Parameter(torch.FloatTensor(1, self.npoint, 2))
        self.pdf.data.uniform_(0, 1)
        self.register_parameter("pdf", self.pdf)
        # ----------------------------------------------------------------------

    def parameterization(self, input, code):

        zeros = torch.zeros(input.size()).cuda()
        ones = torch.ones(input.size()).cuda()
        noise = torch.normal(mean=zeros, std=ones * self.std)
        input = input + noise

        return self.map2Dto3D(input, code)

    def chart(self, input, code):

        return self.map6Dto2D(input, code)
