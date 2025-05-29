"""
Copyright (C) 2025 Beihang University, China
SPDX-License-Identifier: Apache-2.0
Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
Author: Haiyang Long
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import csv
import networkx as nx

class AdjustableSpatialAttention(nn.Module):
    """可调整层数的改进空间注意力模块，用于捕获EEG电极间的复杂空间关系"""
    def __init__(self, num_channels, channel_layers=3, spatial_layers=2, reduction_ratio=4):
        super().__init__()
        self.channel_attention = self._build_channel_attention(num_channels, channel_layers, reduction_ratio)
        self.spatial_conv = self._build_spatial_conv(num_channels, spatial_layers)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(num_channels * 2, num_channels, kernel_size=1),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True)
        )
        
        self._reset_parameters()

    def _build_channel_attention(self, num_channels, layers, reduction_ratio):
        modules = [nn.LayerNorm([num_channels])]
        
        for i in range(layers):
            if i == 0:
                modules.append(nn.Linear(num_channels, num_channels // reduction_ratio))
            elif i == layers - 1:
                modules.append(nn.Linear(num_channels // reduction_ratio, num_channels))
            else:
                modules.append(nn.Linear(num_channels // reduction_ratio, num_channels // reduction_ratio))
            
            if i < layers - 1:
                modules.append(nn.ReLU(inplace=True))
        
        modules.append(nn.Sigmoid())
        return nn.Sequential(*modules)

    def _build_spatial_conv(self, num_channels, layers):
        modules = []
        for _ in range(layers):
            modules.extend([
                nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(num_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*modules)

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, channels, time]
        identity = x
        
        channel_weights = self.channel_attention(x.mean(dim=2))
        channel_weights = channel_weights.unsqueeze(-1)
        
        spatial = self.spatial_conv(x)
        
        weighted = x * channel_weights
        concatenated = torch.cat([weighted, spatial], dim=1)
        out = self.fusion(concatenated)
        
        return identity + out * 0.1  # 使用残差连接和小的缩放因子

class EnhancedChannelInteraction(nn.Module):
    """增强版通道交互模块，增加了内部层数以捕捉更复杂的通道间关系"""
    def __init__(self, num_channels, groups=4, num_layers=8):
        super().__init__()
        self.groups = groups
        self.num_layers = num_layers
        
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = num_channels
            out_channels = num_channels
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, groups=min(self.groups, in_channels)),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(num_channels, max(num_channels // 16, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(num_channels // 16, 1), num_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self._reset_parameters()
        
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        identity = x
        out = x
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        se_weight = self.se(out)
        out = out * se_weight
        return identity + torch.sigmoid(self.alpha) * out

class EEGClassifier(nn.Module):
    """改进的EEG分类器主模块，整合可调整的空间注意力和增强的通道交互机制"""
    def __init__(self, num_classes, input_size=(2000, 31), spatial_attention_layers=(8, 4)):
        super().__init__()
        
        self.spatial_attention = AdjustableSpatialAttention(input_size[1], 
                                                            channel_layers=spatial_attention_layers[0], 
                                                            spatial_layers=spatial_attention_layers[1])

        self.features = nn.Sequential(                                                                                                                                                                                                                                                                                                                                      
            nn.Conv1d(input_size[1], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.channel_interaction = nn.Sequential(
            EnhancedChannelInteraction(64, num_layers=4),
            EnhancedChannelInteraction(64, num_layers=4),
            EnhancedChannelInteraction(64, num_layers=4)
        )
        
        self.deep_features = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            EnhancedChannelInteraction(128, num_layers=4),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            EnhancedChannelInteraction(256, num_layers=4),
            
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            EnhancedChannelInteraction(512, num_layers=4)
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(8)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.spatial_attention(x)
        x = self.features(x)
        x = self.channel_interaction(x)
        x = self.deep_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
