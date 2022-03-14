# Design and implementation of liver organ segmentation based on small samples

Based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), add generic_Ours.py and transformer_block.py.

- Change the last two layer encoder part of nnU-Net to Transformer structure. The added vision Transformer structure is based on [Segformer](http://arxiv.org/abs/2105.15203).

- Add [Shifted Patch Tokenization](http://arxiv.org/abs/2112.13492) optimization in nnU-Net and apply it to patch embedding.

Please refer to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for configuration.

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN
- nnU-Net frame