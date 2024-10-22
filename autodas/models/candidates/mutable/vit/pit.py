# Copyright 2021-present NAVER Corp.
# Apache License v2.0
"""
Modified from the official implementation of PiT.
https://github.com/naver-ai/pit/blob/master/pit.py
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

from .base import BaseTransformerModel
from .common import TransformerLayer, layernorm
from .config import cfg


class Transformer(nn.Module):

    def __init__(self,
                 embed_dim,
                 depth,
                 heads,
                 mlp_ratio,
                 drop_rate=.0,
                 attn_drop_rate=.0,
                 drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerLayer(
                in_channels=embed_dim,
                num_heads=heads,
                qkv_bias=True,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_prob[i]) for i in range(depth)
        ])

    def forward(self, x, cls_tokens):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')

        token_length = cls_tokens.shape[1]
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            blk.shape_info = (token_length, h, w)
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x, cls_tokens


class conv_head_pooling(nn.Module):

    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(
            in_feature,
            out_feature,
            kernel_size=stride + 1,
            padding=stride // 2,
            stride=stride,
            padding_mode=padding_mode,
            groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Module):

    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PIT(BaseTransformerModel):

    def __init__(self, arch_config=None, num_classes=None):
        super(PIT, self).__init__()
        self.stride = cfg.PIT.STRIDE

        if arch_config:
            self.base_dim = arch_config['base_dim']
            self.depth = arch_config['depth']
            self.num_heads = arch_config['num_heads']
            self.mlp_ratio = arch_config['mlp_ratio']
        else:
            self.base_dim = cfg.PIT_SUBNET.BASE_DIM
            self.depth = cfg.PIT_SUBNET.DEPTH
            self.num_heads = cfg.PIT_SUBNET.NUM_HEADS
            self.mlp_ratio = cfg.PIT_SUBNET.MLP_RATIO

        if num_classes:
            self.num_classes = num_classes
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES

        self.hidden_dim = [
            self.base_dim * self.num_heads[i] for i in range(len(self.depth))
        ]
        self.feature_dims = sum([[self.hidden_dim[i]] * d
                                 for i, d in enumerate(self.depth)], [])

        total_block = sum(self.depth)
        block_idx = 0
        embed_size = math.floor((self.img_size - self.patch_size) /
                                self.stride + 1)

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.base_dim * self.num_heads[0], embed_size,
                        embed_size),
            requires_grad=True)
        self.patch_embed = conv_embedding(self.in_channels,
                                          self.base_dim * self.num_heads[0],
                                          self.patch_size, self.stride, 0)

        self.num_tokens = 1

        self.cls_token = nn.Parameter(
            torch.randn(1, self.num_tokens, self.base_dim * self.num_heads[0]))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(self.depth)):
            drop_path_prob = [
                self.drop_path_rate * i / total_block
                for i in range(block_idx, block_idx + self.depth[stage])
            ]
            block_idx += self.depth[stage]

            self.transformers.append(
                Transformer(self.base_dim * self.num_heads[stage],
                            self.depth[stage], self.num_heads[stage],
                            self.mlp_ratio, self.drop_rate,
                            self.attn_drop_rate, drop_path_prob))
            if stage < len(self.depth) - 1:
                self.pools.append(
                    conv_head_pooling(
                        self.base_dim * self.num_heads[stage],
                        self.base_dim * self.num_heads[stage + 1],
                        stride=2))

        layers = [[m for m in t.blocks] for t in self.transformers]
        layers = sum(layers, [])
        self.initialize_hooks(layers)

        self.norm = layernorm(self.base_dim * self.num_heads[-1])
        self.embed_dim = self.base_dim * self.num_heads[-1]

        self.head = nn.Linear(self.base_dim * self.num_heads[-1],
                              self.num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _feature_hook(self, module, inputs, outputs):
        token_length, h, w = module.shape_info
        x = outputs[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        self.features.append(x)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def forward(self, x, is_feat=False):
        cls_token = self.forward_features(x)
        x_cls = self.head(cls_token[:, 0])

        if is_feat:
            return self.features, x_cls
        else:
            return x_cls


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    test_config = {
        'mlp_ratio': 6,
        'num_heads': [4, 4, 8],
        'base_dim': 40,
        'depth': [1, 6, 6]
    }
    net = PIT(arch_config=test_config, num_classes=100)
    feats, logit = net(x, is_feat=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
