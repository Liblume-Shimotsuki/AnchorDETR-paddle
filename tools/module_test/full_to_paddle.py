import numpy as np
import torch
import paddle

def torch2paddle(torch_path, paddle_path):
    # torch_path = "/home/lazurite/share_weight/transformer.pth"
    # paddle_path = "/home/lazurite/share_weight/transformer.pdparams"
    torch_state_dict = torch.load(torch_path)["model"]
    fc_names = ["adapt_pos1d", "adapt_pos2d", "ffn", "class_embed", "bbox_embed"]
    paddle_state_dict = {}
    for k in torch_state_dict:
        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        ### decoder 部分的self-attantion层
        if "decoder_layers" in k and "self_attn" in k and "weight" in k:
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)
        flag = [i in k for i in fc_names]
        if any(flag) and ".weight" in k and "norm" not in k: # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # if k not in model_state_dict:
        if False:
            print(k)
        else:
            paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)

if __name__ == "__main__":
    torch2paddle("/home/lazurite/clone/AnchorDETR/AnchorDETR_r50_dc5.pth", "/home/lazurite/clone/AnchorDETR/AnchorDETR_r50_dc5.pdparams")