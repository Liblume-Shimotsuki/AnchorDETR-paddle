import paddle
import torch
from ppdet.modeling import DETRTransformer, AnchorDETRTransformer
from ppdet.modeling.heads.detr_head import AnchorDETRHead
from torch2paddle import torch2paddle

detr_transformer = AnchorDETRTransformer()
transformer = detr_transformer.transformer

head = AnchorDETRHead(
    num_classes=91,
    fpn_dims=[],
    use_focal_loss=True
)

torch2paddle("/home/lazurite/share_weight/transformer_dc_init.pth", "/home/lazurite/share_weight/transformer_dc_init.pdparams")
transformer_weight = paddle.load("/home/lazurite/share_weight/transformer_dc_init.pdparams")
transformer.load_dict(transformer_weight)

for key in transformer.state_dict().keys():
    if key in transformer_weight.keys():
        shape_0 = transformer.state_dict()[key].shape
        shape_1 = transformer_weight[key].shape
        if shape_0 != shape_1:
            print(">>>", key, shape_0, shape_1)
for key in transformer_weight.keys():
    if key in transformer.state_dict().keys():
        shape_0 = transformer.state_dict()[key].shape
        shape_1 = transformer_weight[key].shape
        if shape_0 != shape_1:
            print(">>>", key, shape_0, shape_1)

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

for key in head.state_dict().keys():
    if key in head_weight.keys():
        shape_0 = head.state_dict()[key].shape
        shape_1 = head_weight[key].shape
        if shape_0 != shape_1:
            print(">>>", key, shape_0, shape_1)
for key in head_weight.keys():
    if key in head.state_dict().keys():
        shape_0 = head.state_dict()[key].shape
        shape_1 = head_weight[key].shape
        if shape_0 != shape_1:
            print(">>>", key, shape_0, shape_1)


torch2paddle("/home/lazurite/share_weight/input_proj_dc_init.pth", "/home/lazurite/share_weight/input_proj_dc_init.pdparams")
input_proj_weight = paddle.load("/home/lazurite/share_weight/input_proj_dc_init.pdparams")
detr_transformer.input_proj.load_dict(input_proj_weight)

paddle.save(detr_transformer.state_dict(), "./detr_transformer_dc_init.pdmodel")
paddle.save(head.state_dict(), "./detr_head_dc_init.pdmodel")
