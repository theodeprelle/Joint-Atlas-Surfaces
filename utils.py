
import os
import json
import argparse
import random
import tqdm
import numpy as np


tqdm_format = '{desc:40} : {percentage:3.0f}%|{bar}{r_bar}'

def save_params(settings, parameters, loss):

    path = os.path.join('Log', settings.exp, settings.name, 'settings.json')
    settings_json_file = open(path, 'w')
    json.dump(settings.__dict__, settings_json_file)
    settings_json_file.close()

    path = os.path.join('Log', settings.exp, settings.name,
                        'training_losses.json')
    loss_json_file = open(path, 'w')
    json.dump(loss, loss_json_file)
    loss_json_file.close()

    for parameters_name in parameters:

        path = os.path.join('Log', settings.exp,
                            settings.name, parameters_name + '.param')

        if hasattr(parameters[parameters_name], 'state_dict'):
            torch.save(parameters[parameters_name].state_dict(), path)
        else:
            torch.save(parameters[parameters_name], path)


def parse_args(inference=False):
    """
    Training argument loader

    Input :
        - inference : Boolean, if set to True then it load existing settings

    Output :
        - settings : Training settings dictionary
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--shape', type=str, default='')
    parser.add_argument('--collection_shape', type=str, default="None")
    parser.add_argument('--exp', type=str, default='test')
    parser.add_argument('--npoint', type=int, default=10000)
    parser.add_argument('--nepoch', type=int, default=150000)
    parser.add_argument('--nlatent', type=int, default=256)
    parser.add_argument('--normal_factor', type=float, default=.01)
    parser.add_argument('--repulsive_factor', type=float, default=1)
    parser.add_argument('--cycle_factor', type=float, default=10)
    parser.add_argument('--sigma_factor', type=float, default=1)
    parser.add_argument('--regul_factor', type=float, default=1)
    parser.add_argument('--point_per_gaussian', type=int, default=1)
    settings = parser.parse_args()

    if inference:
        path = os.path.join('Log',
                            settings.exp,
                            settings.name,
                            'settings.json')

        with open(path, 'r') as settings_path:
            settings = json.load(settings_path)

    return settings


def create_log(settings):
    """
    Create training log folders
    Input :
        - settings : Training settings dictionary
    """

    if not os.path.exists('Log'):
        os.mkdir('Log')

    if not os.path.exists(os.path.join('Log', settings.exp)):
        os.mkdir(os.path.join('Log', settings.exp))

    if not os.path.exists(os.path.join('Log', settings.exp, settings.name)):
        os.mkdir(os.path.join('Log', settings.exp, settings.name))


def read_xyz_format(path):
    pointcloud = []
    with open(path, 'r') as f:
        d = f'Loading the pointcloud : {path}'
        for lines in tqdm.tqdm(f.readlines(), desc=d, bar_format=tqdm_format):
            pointcloud.append(np.array(lines.split(' ')[:-1]).astype(np.float))

    return np.array(pointcloud)


def triangle_area(v1, v2, v3):
    areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    return areas


def save_xyz_file(pointcloud, path):

    data = ''
    tq = tqdm.tqdm(pointcloud,
                   desc='Saving training data',
                   bar_format=tqdm_format)
    for x, y, z, nx, ny, nz in tq:
        data += "%s %s %s %s %s %s\n" % (x, y, z, nx, ny, nz)
    with open(path, 'w') as f:
        f.write(data)


def sample_mesh(vertices, faces, N):
    """
    Sample uniformly point on a mesh

    Input:
        - vertices : array of 3D points of size [K,3]
        - faces : array of index of size [L,3]

    Output:
        - pointcloud : array of size [N,6] of 6D coordinates of the pointcloud
    """

    # --------------------------------------------------------------------------
    # samlping point on the mesh
    # --------------------------------------------------------------------------
    v1_xyz = vertices[faces[:, 0], :3]
    v2_xyz = vertices[faces[:, 1], :3]
    v3_xyz = vertices[faces[:, 2], :3]
    v1_normal = vertices[faces[:, 0], 3:]
    v2_normal = vertices[faces[:, 1], 3:]
    v3_normal = vertices[faces[:, 2], 3:]

    # --------------------------------------------------------------------------
    # computing areas
    # --------------------------------------------------------------------------
    areas = triangle_area(v1_xyz, v2_xyz, v3_xyz)
    probabilities = areas / np.sum(areas)

    # --------------------------------------------------------------------------
    # selecting random faces
    # --------------------------------------------------------------------------
    weighted_random_indices = random.choices(np.arange(len(areas)),
                                             weights=probabilities,
                                             k=int(N))

    v1_xyz = v1_xyz[weighted_random_indices]
    v2_xyz = v2_xyz[weighted_random_indices]
    v3_xyz = v3_xyz[weighted_random_indices]
    v1_normal = v1_normal[weighted_random_indices]
    v2_normal = v2_normal[weighted_random_indices]
    v3_normal = v3_normal[weighted_random_indices]

    # --------------------------------------------------------------------------
    # sampling points on the faces
    # --------------------------------------------------------------------------
    u = np.random.rand(int(N), 1)
    v = np.random.rand(int(N), 1)
    is_a_problem = u + v > 1
    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]
    w = 1 - (u + v)

    xyz = (v1_xyz * u) + (v2_xyz * v) + (v3_xyz * w)
    normal = (v1_normal * u) + (v2_normal * v) + (v3_normal * w)
    pointcloud = np.concatenate((xyz, normal), axis=-1)

    return pointcloud


def read_obj_format(path):

    faces = []
    normals = []
    vertices = []

    with open(path, "r") as file:
        tq = tqdm.tqdm(file.readlines(),
                       desc=f'Loading the mesh',
                       bar_format=tqdm_format)
        for ligne in tq:
            if ligne[0:2] == "v ":
                x, y, z = ligne[2:-1].split(" ")
                vertices.append([float(x), float(y), float(z)])
            if ligne[0:2] == "f ":
                x, y, z = ligne[2:-1].split(" ")
                x = x.split('/')[0]
                y = y.split('/')[0]
                z = z.split('/')[0]
                faces.append([int(x) - 1, int(y) - 1, int(z) - 1])
            if ligne[0:2] == "vn":
                x, y, z = ligne[3:-1].split(" ")
                normals.append([float(x), float(y), float(z)])

    faces = np.array(faces).astype(np.int)
    vertices = np.array(vertices).astype(np.float)

    normals = vertex_from_face_normal(vertices, faces)
    vertices = np.concatenate((vertices, normals), axis=1)

    return vertices, faces


def vertex_from_face_normal(vertices, faces):

    normals = np.zeros(vertices.shape).astype(np.float)
    tq = tqdm.tqdm(faces, bar_format=tqdm_format, desc='Computing normals')
    for face in tq:
        a, b, c = face

        ab = vertices[b] - vertices[a]
        ac = vertices[c] - vertices[a]

        cross_product = np.cross(ab, ac)

        normals[a] += cross_product
        normals[b] += cross_product
        normals[c] += cross_product

    norm = np.sqrt(np.sum(np.power(normals, 2), -1)).reshape(len(normals), 1)
    normals /= norm

    return normals


def save_results(settings, uv, uv_mapped, target):

    # --------------------------------------------------------------------------
    # paths
    # --------------------------------------------------------------------------
    path_uv = os.path.join('Log', settings.exp,
                           settings.name, settings.name + '_uv.xyz')
    path_xyz = os.path.join('Log', settings.exp, settings.name,
                            settings.name + '_points.xyz')
    path_target = os.path.join(
        'Log', settings.exp, settings.name, settings.name + '_target.xyz')

    # --------------------------------------------------------------------------
    # target
    # --------------------------------------------------------------------------
    data = ''
    tq = tqdm.tqdm(target, desc='Saving target data', bar_format=tqdm_format)
    for x, y, z, nx, ny, nz in tq:
        data += "%s %s %s %s %s %s\n" % (x, y, z, nx, ny, nz)
    with open(path_target, 'w') as f:
        f.write(data)

    # --------------------------------------------------------------------------
    # prediction
    # --------------------------------------------------------------------------
    data = ''
    tq = tqdm.tqdm(uv_mapped,
                   desc='Saving prediction data',
                   bar_format=tqdm_format)
    for x, y, z, nx, ny, nz in tq:
        data += "%s %s %s %s %s %s\n" % (x, y, z, nx, ny, nz)
    with open(path_xyz, 'w') as f:
        f.write(data)

    # --------------------------------------------------------------------------
    # prediction
    # --------------------------------------------------------------------------
    print('Saving uv data')
    data = ''
    tq = tqdm.tqdm(uv, desc='Saving uv data', bar_format=tqdm_format)
    for u, v in tq:
        data += "%s %s 0\n" % (u, v)
    with open(path_uv, 'w') as f:
        f.write(data)
