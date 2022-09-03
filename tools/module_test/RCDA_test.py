from ppdet.modeling.transformers.rcda import MultiheadRCDA

#
# Test RCDA
# PASSED
#
from torch2paddle import torch2paddle

if __name__ == "__main__":
    import paddle
    import torch
    import numpy as np
    import numpy as np

    def gen_random_array(shape=None, seed=0):
        np.random.seed(seed)
        return np.random.uniform(0, 1, shape)

    attention = MultiheadRCDA(embed_dim=256, num_heads=8, dropout=0)
    src = paddle.to_tensor(gen_random_array([1, 50, 50, 256], 0), dtype=paddle.float32)
    posemb_row = paddle.to_tensor(gen_random_array([1, 50, 50, 256], 1), dtype=paddle.float32)
    posemb_col = paddle.to_tensor(gen_random_array([1, 50, 50, 256], 2), dtype=paddle.float32)
    padding_mask = paddle.zeros([1, 50, 50], dtype=paddle.bool)

    # pth_weight = "/home/lazurite/share_weight/attention.pth"
    # pth_data = torch.load(pth_weight)
    # for k in pth_data.keys():
    #     pth_data[k] = pth_data[k].detach().numpy()
    # attention.load_dict(pth_data)
    torch2paddle("/home/lazurite/share_weight/attention.pth", "/home/lazurite/share_weight/attention.pdparams")
    attention.load_dict(paddle.load("/home/lazurite/share_weight/attention.pdparams"))


    res = attention((src+posemb_row).reshape([1, -1, 256]), (src+posemb_col).reshape([1, -1, 256]),
                    src+posemb_row, src+posemb_col,
                    src, key_padding_mask=padding_mask)
    print(res)