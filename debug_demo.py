import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from trainer import trainer_synapse

if __name__ == "__main__":
    vit_name = 'ViT-B_16'
    num_classes = 5
    n_skip = 0
    vit_patches_size = 16
    img_size = 224
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    config_vit.pretrained_path = r'C:\Users\Yangtze\Desktop\TransUNet\ViT-B_16.npz'
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    print('end')
