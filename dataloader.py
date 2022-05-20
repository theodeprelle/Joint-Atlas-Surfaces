import torch
import os
import numpy as np

from utils import read_obj_format, sample_mesh


class SingleShape():
    def __init__(self, settings, mesh_only=False):

        # ----------------------------------------------------------------------
        # parameters of the dataset
        # ----------------------------------------------------------------------
        self.npoint = settings.npoint
        self.nepoch = settings.nepoch

        # ----------------------------------------------------------------------
        # data path
        # ----------------------------------------------------------------------
        path = 'Data/single_shape/'
        mesh_path = os.path.join(path, settings.shape+'.obj')
        pointcloud_path = os.path.join(path, settings.shape+'.npy')

        # ----------------------------------------------------------------------
        # load the input mesh
        # ----------------------------------------------------------------------
        self.vertices, self.faces = read_obj_format(mesh_path)

        # ----------------------------------------------------------------------
        # load or generate the training pointcloud
        # ----------------------------------------------------------------------
        if not os.path.exists(pointcloud_path):

            # ------------------------------------------------------------------
            # sample mesh
            # ------------------------------------------------------------------
            print('Generating training points...')
            self.pointcloud = sample_mesh(self.vertices, self.faces, 2e7)

            # ------------------------------------------------------------------
            # save training data
            # ------------------------------------------------------------------
            np.save(pointcloud_path, self.pointcloud)

        else:

            # ------------------------------------------------------------------
            # read xyz file
            # ------------------------------------------------------------------
            print('Loading training points...')
            self.pointcloud = np.load(pointcloud_path)

        # ----------------------------------------------------------------------
        # random permutation
        # ----------------------------------------------------------------------
        self.pointcloud = self.pointcloud[torch.randperm(len(self.pointcloud))]

        # ----------------------------------------------------------------------
        # normalisation
        # ----------------------------------------------------------------------
        self.mean = np.mean(self.pointcloud[:, :3], 0)
        self.pointcloud[:, :3] -= self.mean
        self.vertices[:, :3] -= self.mean

        self.max = np.max(np.linalg.norm(self.pointcloud[:, :3], axis=1))
        self.vertices[:, :3] /= self.max
        self.pointcloud[:, :3] /= self.max

        n = np.linalg.norm(self.pointcloud[:, 3:], axis=1)
        n = n.reshape(len(n), 1)
        self.pointcloud[:, 3:] /= n
        self.pointcloud[:, 3:] *= settings.normal_factor

        n = np.linalg.norm(self.vertices[:, 3:], axis=1)
        n = n.reshape(len(n), 1)
        self.vertices[:, 3:] /= n
        self.vertices[:, 3:] *= settings.normal_factor

        self.pointcloud = self.pointcloud[0:100000]

    def __getitem__(self, index):

        return torch.FloatTensor(self.pointcloud)

    def get_mesh(self):

        return self.vertices, self.faces

    def get_point_cloud(self):

        return self.pointcloud

    def __len__(self):
        return 1


class Collection():
    def __init__(self, settings, mesh_only=False):

        self.shapes = []
        self.mesh_faces = []
        self.mesh_vertices = []
        self.nepoch = settings.nepoch

        if settings.collection_shape == 'airplane':
            shape = ['plane_'+str(i) for i in range(4)]
        elif settings.collection_shape == 'teddy':
            shape = ['teddy_'+str(i+1) for i in range(4)]
        elif settings.collection_shape == 'ant':
            shape = ['ant_'+str(i+1) for i in range(4)]
        elif settings.collection_shape == 'cups':
            shape = ['cups_'+str(i+1) for i in range(4)]
        elif settings.collection_shape == 'arma':
            shape = ['arma_'+str(i+1) for i in range(4)]
        else:
            print('wrong collection')
            exit()

        for s in shape:
            print('\nLoading shapes', s)
            settings.shape = s
            dataloader = SingleShape(settings)
            self.shapes.append(dataloader.__getitem__(0))
            v, f = dataloader.get_mesh()
            self.mesh_vertices.append(v)
            self.mesh_faces.append(f)
        self.sampler = RandomSampler(settings, len(self.shapes[0]))

    def __getitem__(self, index):

        if index % 1000 == 0:
            self.permute()
        index = index % len(self.shapes)
        random_index = self.sampler[0]
        return index, self.shapes[index][random_index]

    def permute(self):

        for i in range(len(self.shapes)):
            index = torch.randperm(len(self.shapes[i]))
            self.shapes[i] = self.shapes[i][index]

    def get_point_cloud(self):
        return self.shapes

    def get_mesh(self):

        return self.mesh_vertices, self.mesh_faces

    def __len__(self):
        return (self.nepoch + 1) * len(self.shapes)


class RandomSampler():

    def __init__(self, settings, N):

        # ----------------------------------------------------------------------
        # parameters of the dataset
        # ----------------------------------------------------------------------
        self.npoint = settings.npoint
        self.nepoch = settings.nepoch
        self.N = N

    def __getitem__(self, index):

        start = np.random.randint(0, self.N - 1 - self.npoint)
        end = start + self.npoint
        return np.arange(start, end)

    def __len__(self):
        return self.nepoch + 1


class Toy():
    def __init__(self, settings):

        self.npoint = settings.npoint
        self.nepoch = settings.nepoch

        self.y = torch.zeros(settings.npoint, 6)
        self.y[:, 5] = torch.ones(settings.npoint) * settings.normal_factor

    def __getitem__(self, index):

        self.y[:, :2] = torch.rand(self.npoint, 2)

        return self.y

    def __len__(self):
        return self.nepoch
