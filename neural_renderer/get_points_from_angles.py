from __future__ import division
import math

import torch
'''
azimuth 方位角、elevation 标高、degrees 度数、radians 弧度
math.radians()将角度转换成弧度、math.cos()返回弧度的余弦值
torch.stack()拼接函数。沿着一个新维度对输入张量进行连接。序列中的所有张量都应该为相同形状。
'''
def get_points_from_angles(distance, elevation, azimuth, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        if degrees:
            elevation = math.pi/180. * elevation
            azimuth = math.pi/180. * azimuth
    #
        return torch.stack([
            distance * torch.cos(elevation) * torch.sin(azimuth),
            distance * torch.sin(elevation),
            -distance * torch.cos(elevation) * torch.cos(azimuth)
            ]).transpose(1,0)
