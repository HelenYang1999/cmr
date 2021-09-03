import unittest

import torch

import neural_renderer as nr

class TestLighting(unittest.TestCase):
    
    def test_case1(self):
        """Test whether it is executable."""
        #生成随机数张量，参数定义了维度
        #faces (torch.Tensor): Faces. The shape is [batch size, number of faces, 3 (vertices), 3 (XYZ)].
        #textures (torch.Tensor): Textures. The shape is [batch size, number of faces, texture size, texture size, texture size, 3 (RGB)].
        faces = torch.randn(64, 16, 3, 3, dtype=torch.float32)
        textures = torch.randn(64, 16, 8, 8, 8, 3, dtype=torch.float32)
        nr.lighting(faces, textures)

if __name__ == '__main__':
    unittest.main()



