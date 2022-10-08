# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):#embed_dim = 1024 是位置的最后一维 gridSize是每个小patch的长宽 也就是14
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)#14:0-13
    grid_w = np.arange(grid_size, dtype=np.float32)#14:0-13
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first 生成网格点坐标矩阵。X:[0-13]X14 Y:[0,..][1,..]...[13..]
    grid = np.stack(grid, axis=0)#将两个分离的数据堆叠到一个数据里   #  生成了两个网格。 每个都是14*14  grid现在是（2,14,14）

    grid = grid.reshape([2, 1, grid_size, grid_size])#2x1x14x14
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h,use half of dimensions to encode grid_W
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2) 196x512
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D) 196X1024
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0 #512
    omega = np.arange(embed_dim // 2, dtype=np.float)#0~255 对维度进行遍历
    omega /= embed_dim / 2.#归一化 0-1
    omega = 1. / 10000**omega  # (D/2,) ##有点像做了个反向 本来是0到1 现在是1到0

    pos = pos.reshape(-1)  # (M,) 形式是0到13循环14次
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product 这里是外积 就是一列乘一行 196x256

    emb_sin = np.sin(out) # (M, D/2) 196x256
    emb_cos = np.cos(out) # (M, D/2) 196x256

    emb = np.concatenate([emb_sin, emb_cos], axis=1)#196x512  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']#1x197x768
        embedding_size = pos_embed_checkpoint.shape[-1]#768
        num_patches = model.patch_embed.num_patches#196
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches#1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)#14
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)#14
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
