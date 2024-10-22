import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from .base import BaseTransformerModel
from .common import PatchEmbedding, TransformerLayer, layernorm
from .config import cfg


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class AutoFormer(BaseTransformerModel):

    def __init__(self, arch_config=None, num_classes=None):
        super(AutoFormer, self).__init__()
        # the configs of super arch

        if arch_config:
            self.num_heads = arch_config['num_heads']
            self.mlp_ratio = arch_config['mlp_ratio']
            self.hidden_dim = arch_config['hidden_dim']
            self.depth = arch_config['depth']

        else:
            self.num_heads = cfg.AUTOFORMER_SUBNET.NUM_HEADS
            self.mlp_ratio = cfg.AUTOFORMER_SUBNET.MLP_RATIO
            self.hidden_dim = cfg.AUTOFORMER_SUBNET.HIDDEN_DIM
            self.depth = cfg.AUTOFORMER_SUBNET.DEPTH

        if num_classes:
            self.num_classes = num_classes
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES

        # print('hidden dim is:'. self.hidden_dim)
        self.feature_dims = [self.hidden_dim] * self.depth

        self.patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            out_channels=self.hidden_dim)
        self.num_patches = self.patch_embed.num_patches
        self.num_tokens = 1

        self.blocks = nn.ModuleList()
        dpr = [
            x.item()
            for x in torch.linspace(0, self.drop_path_rate, self.depth)
        ]  # stochastic depth decay rule

        for i in range(self.depth):
            self.blocks.append(
                TransformerLayer(
                    in_channels=self.hidden_dim,
                    num_heads=self.num_heads[i],
                    qkv_bias=True,
                    mlp_ratio=self.mlp_ratio[i],
                    drop_rate=self.drop_rate,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_path_rate=dpr[i],
                ))

        self.initialize_hooks(self.blocks)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + self.num_tokens,
                        self.hidden_dim))
        self.pe_dropout = nn.Dropout(p=self.drop_rate)
        trunc_normal_(self.pos_embed, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = layernorm(self.hidden_dim)

        # classifier head
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

        self.apply(self._init_weights)

    def _feature_hook(self, module, inputs, outputs):
        feat_size = int(self.patch_embed.num_patches**0.5)
        x = outputs[:, self.num_tokens:].view(
            outputs.size(0), feat_size, feat_size, self.hidden_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        self.features.append(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, is_feat=False):
        x = self.patch_embed(x)

        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)

        x = self.pe_dropout(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.head(x[:, 0])

        if is_feat:
            return [self.features[len(self.features) // 2],
                    self.features[-1]], logits
        else:
            return logits


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    test_config = {
        'mlp_ratio':
        [4.0, 3.5, 4.0, 4.0, 3.5, 3.5, 4.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0],
        'num_heads': [3, 4, 3, 4, 4, 4, 4, 3, 4, 4, 4, 4, 3],
        'hidden_dim':
        192,
        'depth':
        13
    }
    net = AutoFormer(arch_config=test_config, num_classes=100)
    feats, logit = net(x, is_feat=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
