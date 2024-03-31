# ViT: PyTorch Paper Replicating

This repository is a PyTorch implementation of the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Alexey Dosovitskiy et al. The paper introduces a new architecture called Vision Transformer (ViT) that applies the transformer to image recognition. The model achieves competitive results on ImageNet and other image recognition benchmarks while being more data-efficient.

## Model Architecture

The model architecture is shown below. The input image is divided into fixed-size non-overlapping patches, which are then linearly embedded. The resulting sequence of embeddings is processed by a transformer encoder, which outputs a sequence of embeddings. The first token of the output sequence is used as the representation of the image, which is then passed through a feedforward network to produce the final output.

![ViT Architecture](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/08-vit-paper-figure-1-architecture-overview.png)

The link to the paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
