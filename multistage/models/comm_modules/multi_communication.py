

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from multistage.data_utils.pre_processor import build_preprocessor


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]
        return x


class Communication(nn.Module):
    def __init__(self, args, confidence_dim):
        super(Communication, self).__init__()
        
        self.smooth = False
        self.lidar_range = args['lidar_range']
        self.voxel_size_x, self.voxel_size_y, self.voxel_size_z = args['voxel_size']
        self.feature_stride = args['feature_stride']
        self.save_percentage = args['save_percentage']
        self.pre_processor = build_preprocessor(args['preprocess'], train=args['train'])
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(confidence_dim, confidence_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(confidence_dim, kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False

    def init_gaussian_filter(self, confidence_dim, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        gaussian_kernel = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.weight.data = gaussian_kernel.repeat(confidence_dim, confidence_dim, 1, 1)
        self.gaussian_filter.bias.data.zero_()

    def regroup(self, confidence_map, cav_nums):
        cum_sum_len = torch.cumsum(cav_nums, dim=0)
        split_map = torch.tensor_split(confidence_map, cum_sum_len[:-1].cpu())
        return split_map

    def forward(self, batch_confidence_maps, cav_nums, tau):
        B = len(cav_nums)
        feature_masks = []
        proposal_masks = []
        comm_rate_feature_list = []
        for b in range(B):
            ori_communication_maps = batch_confidence_maps[b].sigmoid()
            
            topk_mask = self.get_topk(ori_communication_maps)

            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            mask = F.gumbel_softmax(communication_maps, tau=tau, hard=True, dim=1)
            final_mask = topk_mask * mask
            mask_bbx = final_mask[:, 1, :, :]
            mask_feature = final_mask[:, 0, :, :]

            mask_bbx = mask_bbx.unsqueeze(1)
            mask_feature = mask_feature.unsqueeze(1)
        
            comm_rate_feature = mask_feature[0].sum()

            mask_feature_clone = mask_feature.clone()
            mask_bbx_clone = mask_bbx.clone()  

            mask_feature_clone[0] = 1
            mask_bbx_clone[0] = 1

            feature_masks.append(mask_feature_clone)           
            proposal_masks.append(mask_bbx_clone)
            comm_rate_feature_list.append(comm_rate_feature)
        
        comm_feature = sum(comm_rate_feature_list) / B

        return feature_masks, proposal_masks, comm_feature
            
    def get_topk(self, score_maps):
        N, _, H, W = score_maps.shape
        remove_percentage = 1 - self.save_percentage
        mask_list = []
        for n in range(N):
            score_map = score_maps[n]

            feature_score_mask = score_map[0, :, :] >= torch.quantile(score_map[0, :, :].view(-1), remove_percentage, dim=-1, keepdim=True)
            box_score_mask = score_map[1, :, :] >= torch.quantile(score_map[1, :, :].view(-1), remove_percentage, dim=-1, keepdim=True)

            mask_list.append(torch.cat((feature_score_mask.unsqueeze(0), box_score_mask.unsqueeze(0)),dim=0))
        
        return torch.stack(mask_list, dim=0)