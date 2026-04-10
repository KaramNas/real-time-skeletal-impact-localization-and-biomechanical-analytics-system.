import torch
import time
import cv2
import sys
import os
sys.path.insert(0, 'HOT/HOT-main')

from hot.config import cfg
from hot.models import ModelBuilder, SegmentationModule

cfg.merge_from_file('HOT/HOT-main/config/hot-resnet50dilated-c1.yaml')

net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    weights='HOT/HOT-main/ckpt/hot-c1/encoder_epoch_14.pth')
net_decoder = ModelBuilder.build_decoder(
    cfg=cfg,
    arch=cfg.MODEL.arch_decoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights='HOT/HOT-main/ckpt/hot-c1/decoder_epoch_14.pth')

import torch.nn as nn
crit = nn.NLLLoss()
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.cuda()
segmentation_module.eval()

# Simulate a webcam frame
dummy_frame = torch.randn(1, 3, 300, 400).cuda()

# Warmup
with torch.no_grad():
    for _ in range(3):
        _ = segmentation_module.encoder(dummy_frame, return_feature_maps=True)

# Benchmark
times = []
with torch.no_grad():
    for _ in range(10):
        start = time.time()
        _ = segmentation_module.encoder(dummy_frame, return_feature_maps=True)
        torch.cuda.synchronize()
        times.append(time.time() - start)

avg = sum(times) / len(times)
print(f"Average inference time: {avg*1000:.1f}ms")
print(f"Theoretical max FPS: {1/avg:.1f}")