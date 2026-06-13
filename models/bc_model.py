"""BC 控制脑：画面像素 → 手柄按键 + 摇杆。

ResNet-18 做视觉编码器（ImageNet 预训练），双头输出：
- 按钮头：17 键多标签分类
- 摇杆头：6 轴回归

支持 num_frames > 1 的帧堆叠：第一层卷积输入通道从 3 变为 num_frames*3，
其余层保留 ImageNet 预训练权重。
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

NUM_BUTTONS = 17
NUM_AXES = 6


class BCModel(nn.Module):
    def __init__(self, num_buttons=NUM_BUTTONS, num_axes=NUM_AXES,
                 freeze_backbone=False, num_frames=1):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        in_channels = num_frames * 3
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=old_conv.bias is not None)
            # 复制 RGB 权重到每组通道
            with torch.no_grad():
                for c in range(num_frames):
                    new_conv.weight[:, c*3:(c+1)*3] = old_conv.weight
            backbone.conv1 = new_conv

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 512

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.button_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_buttons),
        )

        self.axis_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_axes),
            nn.Tanh(),
        )

    def forward(self, x):
        f = self.encoder(x)
        return self.button_head(f), self.axis_head(f)
