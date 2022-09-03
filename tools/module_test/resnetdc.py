import numpy as np
import paddle
from ppdet.core.workspace import create
from ppdet.modeling.backbones.resnet import ResNet
from paddle import nn

backbone = ResNet(**{'depth': 50,
                     'norm_type': 'bn',
                     'freeze_at': 0,
                     'return_idx': [3],
                     'lr_mult_list': [0.1, 0.1, 0.1, 0.1],
                     'num_stages': 4,
                     'freeze_norm': True})

backbone.res5.res5a.short.conv = nn.Conv2D(1024, 2048, kernel_size=[1, 1], stride=[1, 1])
backbone.res5.res5a.branch2b.conv = nn.Conv2D(512, 512, kernel_size=[3, 3], stride=[1, 1], padding=1)
backbone.res5.res5b.branch2b.conv = nn.Conv2D(512, 512, kernel_size=[3, 3], stride=[1, 1], padding=2, dilation=2)
backbone.res5.res5c.branch2b.conv = nn.Conv2D(512, 512, kernel_size=[3, 3], stride=[1, 1], padding=2, dilation=2)


def gen_random_array(shape=None, seed=0):
    np.random.seed(seed)
    return np.random.uniform(0, 1, shape)

src = paddle.to_tensor(gen_random_array([1, 3, 800, 800], 0), dtype=paddle.float32)
input = {"image": src}
output = backbone(input)
print(output[-1].shape)
print(output)