# Swin Transformer Assisted Prior Attention Network for Skin Lesion Segmentation

This is the implementation of ['Swin Transformer Assisted Prior Attention Network for Skin Lesion Segmentation'](https://www.mdpi.com/2076-3417/12/9/4735)  which is accepted by Applied Sciences.

![framework](https://github.com/Frank-Star-fn/Swin-PANet/blob/main/pics/framework.png)


In this paper, a single Transformer layer network based on traditional encoder-decoder structure, named SAT-Net, is proposed which employs Transformer to bridge the inconsistent information between encoder and decoder. What’s more, an accelerated convergence module and a feature attention enhancement module are proposed to reduce memory and computation consumption and improve attention capability of network. Extensive experiments are conducted to validate the effectiveness and advantages of the proposed network on three skin lesion segmentation datasets, including ISIC 2016, ISIC 2018 and PH2 datasets. Ablation studies demonstrate the effectiveness of the memory-efficient and computation-efficient network. Comparisons with state-of-the-art methods also demonstrate that the proposed network outperforming its peers.

## Code List

- [x] Network
- [ ] Training Codes
- [ ] Testing Codes
- [ ] Evaluation

## Usage

### 1. Data Preparation
#### ISIC, MoNuSeg and GlaS Datasets

First, you can download the dataset from following links:

* ISIC 2016 Dataset - [Link (Original)](https://challenge.isic-archive.com/data/)
* MoNuSeg Dataset - [Link (Original)](https://monuseg.grand-challenge.org/Data/)
* GLAS Dataset - [Link (Original)](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest)


### 2. Training


### 3. Testing and Evaluation



## Reference

* [UCTransNet](https://github.com/McGregorWwww/UCTransNet) 
* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) 
* [TransUNet](https://github.com/Beckschen/TransUNet) 
* [MedT](https://github.com/jeya-maria-jose/Medical-Transformer)



