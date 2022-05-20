import torch


def compute_jac(x, x_mapped, INDEX):

    # --------------------------------------------------------------------------
    # compute the gradient
    # --------------------------------------------------------------------------
    j_x = torch.autograd.grad(outputs=x_mapped,
                              inputs=x,
                              grad_outputs=INDEX[0],
                              create_graph=True)
    j_y = torch.autograd.grad(outputs=x_mapped,
                              inputs=x,
                              grad_outputs=INDEX[1],
                              create_graph=True)
    j_z = torch.autograd.grad(outputs=x_mapped,
                              inputs=x,
                              grad_outputs=INDEX[2],
                              create_graph=True)

    # --------------------------------------------------------------------------
    # resize
    # --------------------------------------------------------------------------
    j_x = j_x[0].contiguous()
    j_y = j_y[0].contiguous()
    j_z = j_z[0].contiguous()

    return j_x, j_y, j_z


def jacobian_matrix(jacobian):

    j_x = jacobian[0]
    j_y = jacobian[1]
    j_z = jacobian[2]

    # --------------------------------------------------------------------------
    # manually compute the matrix multiplication of J.JT
    # --------------------------------------------------------------------------
    m_11 = j_x[..., 0] ** 2 + j_y[..., 0]**2 + j_z[..., 0]**2
    m_22 = j_x[..., 1] ** 2 + j_y[..., 1]**2 + j_z[..., 1]**2
    m_21 = j_x[..., 0] * j_x[..., 1] + j_y[..., 0] * \
        j_y[..., 1] + j_z[..., 0] * j_z[..., 1]

    return m_11, m_22, m_21


def normals_from_jacobian(jacobian):

    j_x = jacobian[0]
    j_y = jacobian[1]
    j_z = jacobian[2]

    # --------------------------------------------------------------------------
    # compute and normalize the normals
    # --------------------------------------------------------------------------
    nx = (j_y[..., 0] * j_z[..., 1] - j_y[..., 1] * j_z[..., 0]).unsqueeze(-1)
    ny = (j_z[..., 0] * j_x[..., 1] - j_z[..., 1] * j_x[..., 0]).unsqueeze(-1)
    nz = (j_x[..., 0] * j_y[..., 1] - j_x[..., 1] * j_y[..., 0]).unsqueeze(-1)

    # --------------------------------------------------------------------------
    # scaling
    # --------------------------------------------------------------------------
    normal = torch.cat((nx, ny, nz), -1)
    norm = torch.norm(normal, dim=-1).detach()
    normal /= (norm.unsqueeze(-1) + 1e-9)
    return normal.contiguous()


def isometric(j_x, j_y, j_z):

    # --------------------------------------------------------------------------
    # manually compute the matrix multiplication of J.JT
    # --------------------------------------------------------------------------
    m_11 = j_x[..., 0] ** 2 + j_y[..., 0]**2 + j_z[..., 0]**2
    m_22 = j_x[..., 1] ** 2 + j_y[..., 1]**2 + j_z[..., 1]**2
    m_21 = j_x[..., 0] * j_x[..., 1] + j_y[..., 0] * \
        j_y[..., 1] + j_z[..., 0] * j_z[..., 1]

    # --------------------------------------------------------------------------
    # conformal loss
    # --------------------------------------------------------------------------
    loss = (m_11 - 1) ** 2 + (m_22 - 1) ** 2 + 2 * m_21 ** 2

    return torch.mean(loss)
