import torch
import paddle

from ppdet.modeling.transformers.anchor_detr_transformer import TransformerInAnchorDETR
from ppdet.modeling.heads.detr_head import AnchorDETRHead
from torch2paddle import torch2paddle

transformer = TransformerInAnchorDETR()
head = AnchorDETRHead(
    num_classes=91,
    fpn_dims=[],
    use_focal_loss=True
)
transformer_input = torch.load("/home/lazurite/share_weight/transformer_input.pth")
srcs = transformer_input["srcs"][0].cpu().detach().numpy()
mask = transformer_input["masks"][0].cpu().detach().numpy()
srcs = paddle.to_tensor(srcs, dtype=paddle.float32)
mask = paddle.to_tensor(mask, dtype=paddle.bool)

torch2paddle("/home/lazurite/share_weight/transformer.pth", "/home/lazurite/share_weight/transformer.pdparams")
transformer_weight = paddle.load("/home/lazurite/share_weight/transformer.pdparams")
transformer.load_dict(transformer_weight)
head_weight = {
    "class_embed.weight": transformer_weight["class_embed.0.weight"],
    "class_embed.bias": transformer_weight["class_embed.0.bias"],
    "bbox_embed.layers.0.weight": transformer_weight["bbox_embed.0.layers.0.weight"],
    "bbox_embed.layers.0.bias": transformer_weight["bbox_embed.0.layers.0.bias"],
    "bbox_embed.layers.1.weight": transformer_weight["bbox_embed.0.layers.1.weight"],
    "bbox_embed.layers.1.bias": transformer_weight["bbox_embed.0.layers.1.bias"],
    "bbox_embed.layers.2.weight": transformer_weight["bbox_embed.0.layers.2.weight"],
    "bbox_embed.layers.2.bias": transformer_weight["bbox_embed.0.layers.2.bias"],
}
head.load_dict(head_weight)

res = transformer(srcs, mask)
res = head(res, None)
print(res)
