# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention, _convert_attention_mask
from .position_encoding import PositionEmbedding
from .utils import _get_clones
from ..initializer import linear_init_, conv_init_, xavier_uniform_, normal_, constant_, uniform_, _calculate_fan_in_and_fan_out
from .rcda import *

__all__ = ['AnchorDETRTransformer']


class TransformerDecoderLayer(nn.Layer):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0., activation="relu", n_heads=8):
        super().__init__()

        # cross attention
        self.cross_attn = MultiheadRCDA(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, srcs, src_padding_masks=None, adapt_pos2d=None,
                adapt_pos1d=None, posemb_row=None, posemb_col=None, posemb_2d=None):
        tgt_len = tgt.shape[1]

        query_pos = adapt_pos2d(pos2posemb2d(reference_points))
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        # This Self-Attention return attn_output only.
        tgt2 = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        bz, l, c, h, w = srcs.shape
        srcs = srcs.reshape([bz * l, c, h, w]).transpose([0, 2, 3, 1])

        query_pos_x = adapt_pos1d(pos2posemb1d(reference_points[..., 0]))
        query_pos_y = adapt_pos1d(pos2posemb1d(reference_points[..., 1]))
        posemb_row = posemb_row.unsqueeze(1).tile([1, h, 1, 1])
        posemb_col = posemb_col.unsqueeze(2).tile([1, 1, w, 1])
        src_row = src_col = srcs
        k_row = src_row + posemb_row
        k_col = src_col + posemb_col
        tgt2 = self.cross_attn((tgt + query_pos_x).tile([l, 1, 1]), (tgt + query_pos_y).tile([l, 1, 1]), k_row, k_col,
                               srcs, key_padding_mask=src_padding_masks)[0].moveaxis(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        return tgt


class TransformerEncoderLayerSpatial(nn.Layer):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0., activation="relu",
                 n_heads=8):
        super().__init__()
        # self attention
        self.self_attn = MultiheadRCDA(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, padding_mask=None, posemb_row=None, posemb_col=None):
        # self attention
        bz, c, h, w = src.shape
        src = src.transpose([0, 2, 3, 1])

        posemb_row = posemb_row.unsqueeze(1).tile([1, h, 1, 1])
        posemb_col = posemb_col.unsqueeze(2).tile([1, 1, w, 1])

        src2 = self.self_attn((src + posemb_row).reshape([bz, h * w, c]), (src + posemb_col).reshape([bz, h * w, c]),
                              src + posemb_row, src + posemb_col,
                              src, key_padding_mask=padding_mask)
        src2 = src2[0].moveaxis(0, 1).reshape([bz, h, w, c])

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.ffn(src)
        src = src.transpose([0, 3, 1, 2])
        return src


class TransformerInAnchorDETR(nn.Layer):
    def __init__(self, hidden_dim=256, nhead=8,
                num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.,
                activation="relu", num_query_position=300, num_query_pattern=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_pattern = num_query_pattern
        self.num_position = num_query_position

        # ================================ Transformer Encoder ================================
        encoder_layer = TransformerEncoderLayerSpatial(
            d_model=hidden_dim,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_heads=nhead)
        self.encoder_layers = _get_clones(encoder_layer, num_encoder_layers)

        # ================================ Transformer Decoder ================================
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_heads=nhead)
        self.decoder_layers = _get_clones(decoder_layer, num_decoder_layers)

        # ================================ Embed ================================
        self.position = nn.Embedding(self.num_position, 2)
        self.pattern = nn.Embedding(self.num_pattern, hidden_dim)

        # ================================ MLP after PostEmb ================================
        self.adapt_pos1d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.adapt_pos2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        uniform_(self.position.weight, 0, 1)
        normal_(self.pattern.weight, 0, 1)
        for layer in self.adapt_pos1d:
            if isinstance(layer, nn.Linear):
                linear_init_(layer)
        for layer in self.adapt_pos2d:
            if isinstance(layer, nn.Linear):
                linear_init_(layer)


    def forward(self, src_proj, mask):
        bs, c, h, w = src_proj.shape
        pos_col, pos_row = mask2pos(mask)
        posemb_row = self.adapt_pos1d(pos2posemb1d(pos_row))
        posemb_col = self.adapt_pos1d(pos2posemb1d(pos_col))
        outputs = src_proj.reshape([bs, c, h, w])
        for lid, layer in enumerate(self.encoder_layers):
            layer: TransformerEncoderLayerSpatial
            outputs = layer(outputs, mask, posemb_row, posemb_col)

        # bs, 1, c, h, w
        srcs = outputs.unsqueeze(1)
        tgt = self.pattern.weight.reshape([1, self.num_pattern, 1, c]). \
            tile([bs, 1, self.num_position, 1]). \
            reshape([bs, self.num_pattern * self.num_position, c])
        reference_points = self.position.weight.unsqueeze(0).tile([bs, self.num_pattern, 1])
        output = tgt

        intermediate_output = []
        for lid, layer in enumerate(self.decoder_layers):
            layer: TransformerDecoderLayer
            output = layer(output, reference_points, srcs, mask,
                           adapt_pos2d=self.adapt_pos2d, adapt_pos1d=self.adapt_pos1d,
                           posemb_row=posemb_row, posemb_col=posemb_col)
            intermediate_output.append(output)
        intermediate_output = paddle.stack(intermediate_output)
        reference = inverse_sigmoid(reference_points)
        return {
            "reference": reference,
            "output": intermediate_output,
        }


@register
class AnchorDETRTransformer(nn.Layer):
    __shared__ = ['hidden_dim']

    def __init__(self,
                 num_queries=100,
                 position_embed_type='sine',
                 return_intermediate_dec=True,
                 backbone_num_channels=2048,
                 hidden_dim=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 num_query_position=300,  # Anchor DETR only
                 num_query_pattern=3,  # Anchor DETR only
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(AnchorDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'

        self.hidden_dim = hidden_dim
        self.nhead = nhead

        self.transformer = TransformerInAnchorDETR(
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_query_position=num_query_position,
            num_query_pattern=num_query_pattern
        )

        # Add a group norm layer
        self.input_proj = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(backbone_num_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim))
        ])

        # ================================ Init Parameters  ================================
        self._reset_parameters()

    def _reset_parameters(self):
        for proj in self.input_proj:
            xavier_uniform_(proj[0].weight, gain=1)
            constant_(proj[0].bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'backbone_num_channels': [i.channels for i in input_shape][-1],
        }

    def forward(self, src, src_mask=None):
        r"""
        Applies a Transformer model on the inputs.

        Parameters:
            src (List(Tensor)): Backbone feature maps with shape [[bs, c, h, w]].
            src_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                [bs, H, W]`. When the data type is bool, the unwanted positions
                have `False` values and the others have `True` values. When the
                data type is int, the unwanted positions have 0 values and the
                others have 1 values. When the data type is float, the unwanted
                positions have `-INF` values and the others have 0 values. It
                can be None when nothing wanted or needed to be prevented
                attention to. Default None.

        Returns:
            output (Tensor): [num_levels, batch_size, num_queries, hidden_dim]
            memory (Tensor): [batch_size, hidden_dim, h, w]
        """
        # use last level feature map
        src_proj = self.input_proj[0](src[-1])  # [B, C, H, W]
        bs, c, h, w = src_proj.shape
        # flatten [B, C, H, W] to [B, HxW, C]
        # src_flatten = src_proj.flatten(2).transpose([0, 2, 1])

        # interpolate mask to match the size of the feature map
        if src_mask is not None:
            src_mask = F.interpolate(
                src_mask.unsqueeze(0).astype(src_proj.dtype),
                size=(h, w))[0].astype('bool')
        else:
            src_mask = paddle.ones([bs, h, w], dtype='bool')

        # to change value to 1 if the pixel is padding otherwise 0.
        mask = ~src_mask

        output = self.transformer(src_proj, mask)
        return output


def mask2pos(mask):
    """
    Convert the not_mask to col and row position.
    not_mask is a tensor with shape [bs, h, w]
    where not_mask[i, j] = False means the pixel at row i and column j is padding.
    where not_mask[i, j] = True means the pixel at row i and column j is real pixel.
    this function is different from the source code of AnchorDETRTransformer implemented in pytorch.
    Args:
        not_mask: [bs, h, w]
    Returns:

    """
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype="float32")
    x_embed = not_mask[:, 0, :].cumsum(1, dtype="float32")
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    return y_embed, x_embed


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    """
    Convert the position to posemb1d.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = paddle.to_tensor(np.arange(num_pos_feats, dtype=np.float32) // 2, dtype=paddle.float32)
    dim_t = temperature ** (2 * dim_t / num_pos_feats)
    pos_x = pos[..., None] / dim_t
    posemb = paddle.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), axis=-1).flatten(-2)
    return posemb


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    # dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = paddle.to_tensor(np.arange(num_pos_feats, dtype=np.float32) // 2, dtype=paddle.float32)
    dim_t = temperature ** (2 * dim_t / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = paddle.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), axis=-1).flatten(-2)
    pos_y = paddle.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), axis=-1).flatten(-2)
    posemb = paddle.concat((pos_y, pos_x), axis=-1)
    return posemb
