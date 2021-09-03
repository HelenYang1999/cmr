from __future__ import division
import math

import torch
#透视
def perspective(vertices, angle=30.):
    '''
    给定一个角度，计算顶点经透视变换后的结果（其实不大懂）
    Compute perspective distortion from a given angle
    '''
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')
    device = vertices.device
    #将角度转换成弧度
    angle = torch.tensor(angle / 180 * math.pi, dtype=torch.float32, device=device)
    angle = angle[None]
    width = torch.tan(angle)
    #[:,None]和[None,:]的区别
    width = width[:, None]
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = torch.stack((x,y,z), dim=2)
    return vertices
