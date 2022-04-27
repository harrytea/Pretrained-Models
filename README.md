# Pretrained-Models

- [News](#news)
- [Vision Transformer](#VisionTransformer)

## News

:smile: **Continue update**

**2022.04.27**

* update vision transformer

## VisionTransformer

You can download the pretrained model from [ViT](https://github.com/rwightman/pytorch-image-models/blob/f55c22bebf9d8afc449d317a723231ef72e0d662/timm/models/vision_transformer.py#L54-L106). Another link is [here](https://console.cloud.google.com/storage/browser/vit_models;tab=objects?prefix=&forceOnObjectsSortingFiltering=false&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))

The official pretrained vit model is `npz` format. You can use `VisionTransformer/jax_pop.py` or  `VisionTransformer/jax_convert.py` to convert it into `pth` file.

```python
python jax_convert.py --src 'xxx.npz' --dst 'xxx.pth'
```

after converting, you can create corresponding model from `VisionTransformer/vit.py` and load the weight. All vit models can be refered to [vision transformer](https://github.com/rwightman/pytorch-image-models/blob/f55c22bebf9d8afc449d317a723231ef72e0d662/timm/models/vision_transformer.py)