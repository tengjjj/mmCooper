import numpy as np
import torch
import torch.nn as nn
from multistage.models.sub_modules.resblock import ResNetModified, BasicBlock
import torch.nn.functional as F

class BEVBackbonePart(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        layer_nums = self.model_cfg['layer_nums']
        layer_strides = self.model_cfg['layer_strides']
        num_filters = self.model_cfg['num_filters']

        assert len(self.model_cfg['layer_nums']) == \
                len(self.model_cfg['layer_strides']) == \
                len(self.model_cfg['num_filters'])

        self.resnet = ResNetModified(BasicBlock, 
                                     layer_nums,
                                    layer_strides,
                                    num_filters)        
        
        # self.fuse_module = AttenFusion(model_cfg['agg_operator']['feature_dim'])

    def forward(self, data_dict):
        spatial_feature = data_dict['spatial_features']

        x = self.resnet(spatial_feature)[0]

        data_dict['spatial_feature_2d'] = x

        return data_dict   