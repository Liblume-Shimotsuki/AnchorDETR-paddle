import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn.functional import softmax, dropout, pad
import copy

from ..initializer import linear_init_, constant_, xavier_uniform_, normal_


def linear(input, weight, bias=None):
    """
    Linear Layer, in pytorch, y = xA^T + b, in paddle, y = xA + b
    Args:
        input:
        weight:
        bias:

    Returns:

    """
    return F.linear(input, weight.T, bias)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def _get_clones(layer, N):
    return nn.LayerList([copy.deepcopy(layer) for i in range(N)])


def multi_head_rcda_forward(query_row,  # type: Tensor
                            query_col,  # type: Tensor
                            key_row,  # type: Tensor
                            key_col,  # type: Tensor
                            value,  # type: Tensor
                            embed_dim_to_check,  # type: int
                            num_heads,  # type: int
                            in_proj_weight,  # type: Tensor
                            in_proj_bias,  # type: Tensor
                            bias_k_row,  # type: Optional[Tensor]
                            bias_k_col,  # type: Optional[Tensor]
                            bias_v,  # type: Optional[Tensor]
                            add_zero_attn,  # type: bool
                            dropout_p,  # type: float
                            out_proj_weight,  # type: Tensor
                            out_proj_bias,  # type: Tensor
                            training=True,  # type: bool
                            key_padding_mask=None,  # type: Optional[Tensor]
                            need_weights=True,  # type: bool
                            attn_mask=None,  # type: Optional[Tensor]
                            use_separate_proj_weight=False,  # type: bool
                            q_row_proj_weight=None,  # type: Optional[Tensor]
                            q_col_proj_weight=None,  # type: Optional[Tensor]
                            k_row_proj_weight=None,  # type: Optional[Tensor]
                            k_col_proj_weight=None,  # type: Optional[Tensor]
                            v_proj_weight=None,  # type: Optional[Tensor]
                            static_k=None,  # type: Optional[Tensor]
                            static_v=None  # type: Optional[Tensor]
                            ):
    bsz, tgt_len, embed_dim = query_row.shape
    src_len_row = key_row.shape[2]
    src_len_col = key_col.shape[1]

    assert embed_dim == embed_dim_to_check
    # assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = 0
    _end = embed_dim
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q_row = linear(query_row, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 1
    _end = embed_dim * 2
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q_col = linear(query_col, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 2
    _end = embed_dim * 3
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k_row = linear(key_row, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 3
    _end = embed_dim * 4
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k_col = linear(key_col, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 4
    _end = None
    _w = in_proj_weight[_start:, :]
    if _b is not None:
        _b = _b[_start:]
    v = linear(value, _w, _b)

    q_row = q_row.moveaxis(0, 1)
    q_col = q_col.moveaxis(0, 1)
    k_row = k_row.mean(1).moveaxis(0, 1)
    k_col = k_col.mean(2).moveaxis(0, 1)

    q_row = q_row * scaling
    q_col = q_col * scaling

    q_row = q_row.reshape([tgt_len, bsz * num_heads, head_dim]).moveaxis(0, 1)
    q_col = q_col.reshape([tgt_len, bsz * num_heads, head_dim]).moveaxis(0, 1)

    if k_row is not None:
        k_row = k_row.reshape([-1, bsz * num_heads, head_dim]).moveaxis(0, 1)
    if k_col is not None:
        k_col = k_col.reshape([-1, bsz * num_heads, head_dim]).moveaxis(0, 1)
    if v is not None:
        v = v.transpose([1, 2, 0, 3]).reshape([src_len_col, src_len_row, bsz * num_heads, head_dim]).transpose(
            [2, 0, 1, 3])

    attn_output_weights_row = paddle.bmm(q_row, k_row.moveaxis(1, 2))
    attn_output_weights_col = paddle.bmm(q_col, k_col.moveaxis(1, 2))
    assert list(attn_output_weights_row.shape) == [bsz * num_heads, tgt_len, src_len_row]
    assert list(attn_output_weights_col.shape) == [bsz * num_heads, tgt_len, src_len_col]

    if key_padding_mask is not None:
        # print(f"shape of key_padding_mask: {key_padding_mask.shape}")
        mask_row = key_padding_mask[:, 0, :].unsqueeze(1).unsqueeze(2)
        mask_col = key_padding_mask[:, :, 0].unsqueeze(1).unsqueeze(2)

        attn_output_weights_row = attn_output_weights_row.reshape([bsz, num_heads, tgt_len, src_len_row])
        attn_output_weights_col = attn_output_weights_col.reshape([bsz, num_heads, tgt_len, src_len_col])

        # attn_output_weights_row = attn_output_weights_row.masked_fill(mask_row,float('-inf'))
        # attn_output_weights_col = attn_output_weights_col.masked_fill(mask_col, float('-inf'))
        # print(f"shape of mask_row: {mask_row.shape}")
        attn_output_weights_row = masked_fill(attn_output_weights_row, mask_row, float('-inf'))
        attn_output_weights_col = masked_fill(attn_output_weights_col, mask_col, float('-inf'))

        attn_output_weights_row = attn_output_weights_row.reshape([bsz * num_heads, tgt_len, src_len_row])
        attn_output_weights_col = attn_output_weights_col.reshape([bsz * num_heads, tgt_len, src_len_col])

    attn_output_weights_col = softmax(attn_output_weights_col, axis=-1)
    attn_output_weights_row = softmax(attn_output_weights_row, axis=-1)

    attn_output_weights_col = dropout(attn_output_weights_col, p=dropout_p, training=training)
    attn_output_weights_row = dropout(attn_output_weights_row, p=dropout_p, training=training)

    efficient_compute = True
    # This config will not affect the performance.
    # It will compute the short edge first which can save the memory and run slightly faster but both of them should get the same results.
    # You can also set it "False" if your graph needs to be always the same.
    if efficient_compute:
        if src_len_col < src_len_row:
            b_ein, q_ein, w_ein = attn_output_weights_row.shape
            b_ein, h_ein, w_ein, c_ein = v.shape
            attn_output_row = paddle.matmul(attn_output_weights_row,
                                            v.transpose([0, 2, 1, 3]).reshape([b_ein, w_ein, h_ein * c_ein])).reshape(
                [b_ein, q_ein, h_ein, c_ein]).transpose([0, 2, 1, 3])
            attn_output = paddle.matmul(attn_output_weights_col.transpose([1, 0, 2])[:, :, None, :],
                                        attn_output_row.transpose([2, 0, 1, 3])).squeeze(-2).reshape(
                [tgt_len, bsz, embed_dim])
            ### the following code base on einsum get the same results
            # attn_output_row = torch.einsum("bqw,bhwc->bhqc",attn_output_weights_row,v)
            # attn_output = torch.einsum("bqh,bhqc->qbc",attn_output_weights_col,attn_output_row).reshape(tgt_len,bsz,embed_dim)
        else:
            b_ein, q_ein, h_ein = attn_output_weights_col.shape
            b_ein, h_ein, w_ein, c_ein = v.shape
            attn_output_col = paddle.matmul(attn_output_weights_col, v.reshape([b_ein, h_ein, w_ein * c_ein])).reshape(
                [b_ein, q_ein, w_ein, c_ein])
            attn_output = paddle.matmul(attn_output_weights_row[:, :, None, :], attn_output_col).squeeze(-2).transpose(
                [1, 0, 2]).reshape([tgt_len, bsz, embed_dim])
            ### the following code base on einsum get the same results
            # attn_output_col = torch.einsum("bqh,bhwc->bqwc", attn_output_weights_col, v)
            # attn_output = torch.einsum("bqw,bqwc->qbc", attn_output_weights_row, attn_output_col).reshape(tgt_len, bsz,embed_dim)
    else:
        b_ein, q_ein, h_ein = attn_output_weights_col.shape
        b_ein, h_ein, w_ein, c_ein = v.shape
        attn_output_col = paddle.matmul(attn_output_weights_col, v.reshape([b_ein, h_ein, w_ein * c_ein])).reshape(
            [b_ein, q_ein, w_ein, c_ein])
        attn_output = paddle.matmul(attn_output_weights_row[:, :, None, :], attn_output_col).squeeze(-2).transpose(
            [1, 0, 2]).reshape([tgt_len, bsz, embed_dim])
        ### the following code base on einsum get the same results
        # attn_output_col = torch.einsum("bqh,bhwc->bqwc", attn_output_weights_col, v)
        # attn_output = torch.einsum("bqw,bqwc->qbc", attn_output_weights_row, attn_output_col).reshape(tgt_len, bsz,embed_dim)

    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        return attn_output, paddle.einsum("bqw,bqh->qbhw", attn_output_weights_row, attn_output_weights_col).reshape(
            tgt_len, bsz, num_heads, src_len_col, src_len_row).mean(2)
    else:
        return attn_output, None


def inverse_sigmoid(x: paddle.Tensor, eps=1e-5):
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1 / x2)


class FFN(nn.Layer):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)


class MultiheadRCDA(nn.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(MultiheadRCDA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = paddle.create_parameter(shape=[5 * embed_dim, embed_dim],
                                                      attr=nn.initializer.XavierUniform(), dtype="float32")
        self.in_proj_bias = paddle.create_parameter(shape=[5 * embed_dim], attr=nn.initializer.Constant(value=0.0),
                                                    dtype="float32")
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=nn.initializer.Constant(value=0.0),
                                  weight_attr=nn.initializer.XavierUniform())
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        linear_init_(self.out_proj)
        constant_(self.out_proj.bias, 0.)

    def forward(self, query_row, query_col, key_row, key_col, value,
                key_padding_mask=None, need_weights=False, attn_mask=None):
        return multi_head_rcda_forward(
            query_row, query_col, key_row, key_col, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            bias_k_row=None, bias_k_col=None, bias_v=None, add_zero_attn=False,
            dropout_p=self.dropout, out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask)
