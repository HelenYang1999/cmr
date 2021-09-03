import torch
import torch.nn.functional as F
import numpy as np

#lighting函数是对纹理进行光影响的叠加，光分为两部分：ambient light环境光 和 directional light平行光
#light的计算公式为：
#intensity_ambient * color_ambient + intensity_directional * ( color_directional * relu(direction * normals) )
def lighting(faces, textures, intensity_ambient=0.5, intensity_directional=0.5,
             color_ambient=(1, 1, 1), color_directional=(1, 1, 1), direction=(0, 1, 0)):

    bs, nf = faces.shape[:2]
    device = faces.device

    # arguments
    # make sure to convert all inputs to float tensors
    if isinstance(color_ambient, tuple) or isinstance(color_ambient, list):
        #torch.tensor从list或者tuple转换成tensor
        color_ambient = torch.tensor(color_ambient, dtype=torch.float32, device=device)
    elif isinstance(color_ambient, np.ndarray):
        color_ambient = torch.from_numpy(color_ambient).float().to(device)
    if isinstance(color_directional, tuple) or isinstance(color_directional, list):
        color_directional = torch.tensor(color_directional, dtype=torch.float32, device=device)
    elif isinstance(color_directional, np.ndarray):
        color_directional = torch.from_numpy(color_directional).float().to(device)
    if isinstance(direction, tuple) or isinstance(direction, list):
        direction = torch.tensor(direction, dtype=torch.float32, device=device)
    elif isinstance(direction, np.ndarray):
        direction = torch.from_numpy(direction).float().to(device)
    #ndimension()，返回tensor的维度
    if color_ambient.ndimension() == 1:
        color_ambient = color_ambient[None, :]
    if color_directional.ndimension() == 1:
        color_directional = color_directional[None, :]
    if direction.ndimension() == 1:
        # None就表示增加一维
        direction = direction[None, :]

    # create light
    light = torch.zeros(bs, nf, 3, dtype=torch.float32).to(device)

    # ambient light
    if intensity_ambient != 0:
        #[1,3] -> [1,1,3]
        temp = color_ambient[:, None, :]
        light += intensity_ambient * color_ambient[:, None, :]

    # directional light
    if intensity_directional != 0:
        #第二维代表顶点
        faces = faces.reshape((bs * nf, 3, 3))
        #faces[:,0]代表取第一维的所有元素，第二维的第一个（也就是第一个顶点），第三维的所有元素
        #那么faces[:, 0] - faces[:, 1]代表的就是batch_size的所有面的第一个顶点坐标 - 第二个顶点的坐标，得到的是[1024,3]
        #顶点0 - 顶点1 得到的是1指向0的向量
        #顶点2 - 顶点1 得到的是1指向2的向量
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]
        # pytorch normalize divides by max(norm, eps) instead of (norm+eps) in chainer
        #cross 是叉积的意思，得到的向量与两者输入都垂直，右手法则
        #F.normalize得到输入的L2正则结果
        normals = F.normalize(torch.cross(v10, v12), eps=1e-5)
        normals = normals.reshape((bs, nf, 3))

        if direction.ndimension() == 2:
            direction = direction[:, None, :]
        cos = F.relu(torch.sum(normals * direction, dim=2))
        # may have to verify that the next line is correct
        temp = color_directional[:, None, :]
        temp = cos[:, :, None]
        light += intensity_directional * (color_directional[:, None, :] * cos[:, :, None])

    # apply
    light = light[:,:,None, None, None, :]
    textures *= light
    return textures
