import argparse
import os.path as osp

import mmcv
import numpy as np
import torch


# stole refactoring code from Robin Strudel, thanks
parser = argparse.ArgumentParser(description='Convert keys from jax official pretrained vit models to ' 'MMSegmentation style.')
parser.add_argument('--src', default='/data4/wangyh/segmenter/segm/tiny.npz', help='src model path or url')
parser.add_argument('--dst', default='/data4/wangyh/segmenter/segm/vit_tiny_p16_384.pth', help='save path')
args = parser.parse_args()



def vit_jax_to_torch(jax_weights, num_layer=12):
    torch_weights = dict()

    # patch embedding
    conv_filters = jax_weights.pop('embedding/kernel')
    conv_filters = conv_filters.permute(3, 2, 0, 1)
    torch_weights['patch_embed.projection.weight'] = conv_filters
    torch_weights['patch_embed.projection.bias'] = jax_weights['embedding/bias']

    # pos embedding
    torch_weights['pos_embed'] = jax_weights['Transformer/posembed_input/pos_embedding']

    # cls token
    torch_weights['cls_token'] = jax_weights['cls']

    # head
    torch_weights['norm.weight'] = jax_weights['Transformer/encoder_norm/scale']
    torch_weights['norm.bias'] = jax_weights['Transformer/encoder_norm/bias']
    torch_weights['head.weight'] = jax_weights['head/kernel']
    torch_weights['head.bias'] = jax_weights['head/bias']

    # transformer blocks
    for i in range(num_layer):
        jax_block = f'Transformer/encoderblock_{i}'
        torch_block = f'layers.{i}'

        # attention norm
        torch_weights[f'{torch_block}.norm1.weight'] = jax_weights[f'{jax_block}/LayerNorm_0/scale']
        torch_weights[f'{torch_block}.norm1.bias'] = jax_weights[f'{jax_block}/LayerNorm_0/bias']

        # attention
        query_weight = jax_weights[f'{jax_block}/MultiHeadDotProductAttention_1/query/kernel']
        query_bias = jax_weights[f'{jax_block}/MultiHeadDotProductAttention_1/query/bias']
        key_weight = jax_weights[f'{jax_block}/MultiHeadDotProductAttention_1/key/kernel']
        key_bias = jax_weights[f'{jax_block}/MultiHeadDotProductAttention_1/key/bias']
        value_weight = jax_weights[f'{jax_block}/MultiHeadDotProductAttention_1/value/kernel']
        value_bias = jax_weights[f'{jax_block}/MultiHeadDotProductAttention_1/value/bias']

        qkv_weight = torch.from_numpy(np.stack((query_weight, key_weight, value_weight), 1))
        qkv_weight = torch.flatten(qkv_weight, start_dim=1)
        qkv_bias = torch.from_numpy(np.stack((query_bias, key_bias, value_bias), 0))
        qkv_bias = torch.flatten(qkv_bias, start_dim=0)

        torch_weights[f'{torch_block}.attn.qkv.weight'] = qkv_weight
        torch_weights[f'{torch_block}.attn.qkv.bias'] = qkv_bias
        to_out_weight = jax_weights[f'{jax_block}/MultiHeadDotProductAttention_1/out/kernel']
        to_out_weight = torch.flatten(to_out_weight, start_dim=0, end_dim=1)
        torch_weights[f'{torch_block}.attn.proj.weight'] = to_out_weight
        torch_weights[f'{torch_block}.attn.proj.bias'] = jax_weights[f'{jax_block}/MultiHeadDotProductAttention_1/out/bias']

        # mlp norm
        torch_weights[f'{torch_block}.norm2.weight'] = jax_weights[f'{jax_block}/LayerNorm_2/scale']
        torch_weights[f'{torch_block}.norm2.bias'] = jax_weights[f'{jax_block}/LayerNorm_2/bias']

        # mlp
        torch_weights[f'{torch_block}.mlp.fc1.weight'] = jax_weights[f'{jax_block}/MlpBlock_3/Dense_0/kernel']
        torch_weights[f'{torch_block}.mlp.fc1.bias'] = jax_weights[f'{jax_block}/MlpBlock_3/Dense_0/bias']
        torch_weights[f'{torch_block}.mlp.fc2.weight'] = jax_weights[f'{jax_block}/MlpBlock_3/Dense_1/kernel']
        torch_weights[f'{torch_block}.mlp.fc2.bias'] = jax_weights[f'{jax_block}/MlpBlock_3/Dense_1/bias']

    # transpose weights
    for k, v in torch_weights.items():
        if 'weight' in k and 'patch_embed' not in k and 'norm' not in k:
            v = v.permute(1, 0)
        torch_weights[k] = v

    return torch_weights


def main():
    jax_weights = np.load(args.src)
    jax_weights_tensor = {}
    for key in jax_weights.files:
        value = torch.from_numpy(jax_weights[key])
        jax_weights_tensor[key] = value
    if 'L_16-i21k' in args.src:
        num_layer = 24
    else:
        num_layer = 12
    torch_weights = vit_jax_to_torch(jax_weights_tensor, num_layer)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(torch_weights, args.dst)


if __name__ == '__main__':
    main()