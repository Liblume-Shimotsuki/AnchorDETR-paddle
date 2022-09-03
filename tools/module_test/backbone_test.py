import numpy as np
import paddle
from ppdet.core.workspace import create
from ppdet.modeling.backbones.resnet import ResNet


def gen_random_array(shape=None, seed=0):
    np.random.seed(seed)
    return np.random.uniform(0, 1, shape)


backbone = ResNet(**{'depth': 50,
                     'norm_type': 'bn',
                     'freeze_at': 0,
                     'return_idx': [3],
                     'lr_mult_list': [0.1, 0.1, 0.1, 0.1],
                     'num_stages': 4,
                     'freeze_norm': True,
                     "dilation": True})

backbone_weight = paddle.load("/home/lazurite/clone/AnchorDETR/AnchorDETR_r50_dc5_init.pdparams")
backbone_weight_reduce = {}
need_weight_keys = backbone.state_dict().keys()
backbone_weight_keys = backbone_weight.keys()
for key in backbone_weight_keys:
    v = backbone_weight[key]
    if "backbone" in key:
        k = key.replace("backbone.body.", "")
        if k == "conv1.weight":
            k = "conv1.conv1.conv.weight"
        if k.startswith("bn1"):
            k = k.replace("bn1", "norm")
            k = "conv1.conv1." + k
        if k.startswith("layer"):
            layer_idx = int(k[5])
            k = "res" + str(layer_idx + 1) + k[6:]
            sub_layer_idx = int(k.split(".")[1])
            sub_layer_name = "res" + str(layer_idx + 1) + chr(ord('a') + sub_layer_idx)
            k = k.split(".")[0] + "." + sub_layer_name + "." + ".".join(k.split(".")[2:])

            # for shor conv
            if sub_layer_idx == 0 and "downsample" in k:
                k = k.replace("downsample.0", "short.conv")
                k = k.replace("downsample.1", "short.norm")
                backbone_weight_reduce[k] = v
                continue
            compoment = k.split(".")[2]
            branch_idx = int(compoment[-1])
            branch_name = "branch2" + chr(ord('a') + branch_idx - 1) + "." + compoment[:-1]
            k = ".".join(k.split(".")[:2]) + "." + branch_name + "." + ".".join(k.split(".")[3:])
            k = k.replace(".bn", ".norm")
        backbone_weight_reduce[k] = v

for key in backbone_weight_reduce.keys():
    # key to load
    if key in need_weight_keys:
        shape_0 = backbone_weight_reduce[key].shape
        shape_1 = backbone.state_dict()[key].shape
        if shape_0 != shape_1:
            print(">>>", key, shape_0, shape_1)
    else:
        print(f">>> {key} not found")

for key in need_weight_keys:
    # key to load
    if key in backbone_weight_reduce.keys():
        shape_0 = backbone_weight_reduce[key].shape
        shape_1 = backbone.state_dict()[key].shape
        if shape_0 != shape_1:
            print(">>>", key, shape_0, shape_1)
    else:
        print(f">>> {key} not found")
backbone.load_dict(backbone_weight_reduce)
paddle.save(backbone.state_dict(), "./backbone_dc_init.pdparams")
src = paddle.to_tensor(gen_random_array([1, 3, 800, 800], 0), dtype=paddle.float32)
input = {"image": src}
output = backbone(input)
print(output)