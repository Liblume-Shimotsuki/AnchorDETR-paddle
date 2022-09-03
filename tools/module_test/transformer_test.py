import torch
import paddle

from ppdet.modeling.transformers.anchor_detr_transformer import TransformerInAnchorDETR
from torch2paddle import torch2paddle

transformer = TransformerInAnchorDETR()
transformer_input = torch.load("/home/lazurite/share_weight/transformer_input.pth")
srcs = transformer_input["srcs"][0].cpu().detach().numpy()
mask = transformer_input["masks"][0].cpu().detach().numpy()
srcs = paddle.to_tensor(srcs, dtype=paddle.float32)
mask = paddle.to_tensor(mask, dtype=paddle.bool)

torch2paddle("/home/lazurite/share_weight/transformer.pth", "/home/lazurite/share_weight/transformer.pdparams")
transformer_weight = paddle.load("/home/lazurite/share_weight/transformer.pdparams")
transformer.load_dict(transformer_weight)

res = transformer(srcs, mask)
print(res)