
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from multistage.models.sub_modules.torch_transformation_utils import warp_affine_simple
from multistage.models.comm_modules.multi_communication import Communication
from multistage.data_utils.pre_processor import build_preprocessor
from multistage.models.fuse_modules.enhance_transformer import RPN_transformer_deformable_mtf

class SingleRedundancyAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SingleRedundancyAttention, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)
        self.query_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x_unfold = F.unfold(x, kernel_size=3, padding=1).view(cav_num, C,9,H*W).permute(3,0,2,1).contiguous().view(H*W, cav_num*9,C)   
        query = self.query_proj(x)
        query = query.view(cav_num, C, -1).permute(2, 0, 1)      
        out = self.att(query, x_unfold, x_unfold)
        out = out.permute(1, 2, 0).view(cav_num, C, H, W)
        return out 

class RedundancyAttention(nn.Module):
    def __init__(self, feature_dim, args):
        super(RedundancyAttention, self).__init__()
        self.multi_att = nn.ModuleList()
        self.layer_num = 2
        self.fuse_layer_num = args['fuse_layer_num']
        for i in range(self.layer_num):
            self.multi_att.append(SingleRedundancyAttention(feature_dim))

    def forward(self, x):
        for i in range(self.fuse_layer_num):
            x = self.multi_att[i](x)
        out = x[0]
        return out


class BoxExcrator(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 use_norm=True,
                 ):
        
        super().__init__()

        self.use_norm = use_norm

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x)
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        return x

class MultiScaledDotProductAttention(nn.Module):
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
        super(MultiScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        hw, cav_num, C = query.shape
        score = torch.einsum('bnd,md->bnm', query, key) / self.sqrt_dim       
        attn = F.softmax(score, -1)
        context = torch.einsum('bnd,dm->bnm', attn, value)
        #context = torch.bmm(attn, value)
        return context
    
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


class EncodeLayer(nn.Module):
    def __init__(self, channels, n_head=8, dropout=0):
        super(EncodeLayer, self).__init__()
        self.attn = nn.MultiheadAttention(channels, n_head, dropout)
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, q, k, v, confidence_map=None):
        """
        order (seq, batch, feature)
        Args:
            q: (1, H*W, C)
            k: (N, H*W, C)
            v: (N, H*W, C)
        Returns:
            outputs: ()
        """
        residual = q
        if confidence_map is not None:
            context, weight = self.attn(q,k,v, quality_map=confidence_map) # (1, H*W, C)
        else:
            context, weight = self.attn(q,k,v) # (1, H*W, C)
        context = self.dropout1(context)
        output1 = self.norm1(residual + context)

        # feed forward net
        residual = output1 # (1, H*W, C)
        context = self.linear2(self.relu(self.linear1(output1)))
        context = self.dropout2(context)
        output2 = self.norm2(residual + context)

        return output2

class TransformerFusion(nn.Module):
    def __init__(self, channels=256, n_head=8, with_spe=True, with_scm=True, dropout=0):
        super(TransformerFusion, self).__init__()

        self.encode_layer = EncodeLayer(channels, n_head, dropout)
        self.with_spe = with_spe
        self.with_scm = with_scm
        
    def forward(self, batch_neighbor_feature, batch_neighbor_feature_pe, batch_confidence_map, record_len):
        x_fuse = []
        B = len(record_len)
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            neighbor_feature = batch_neighbor_feature[b]
            _, C, H, W = neighbor_feature.shape
            neighbor_feature_flat = neighbor_feature.view(N,C,H*W)  # (N, C, H*W)

            if self.with_spe:
                neighbor_feature_pe = batch_neighbor_feature_pe[b]
                neighbor_feature_flat_pe = neighbor_feature_pe.view(N,C,H*W)  # (N, C, H*W)
                query = neighbor_feature_flat_pe[0:1,...].permute(0,2,1)  # (1, H*W, C)
                key = neighbor_feature_flat_pe.permute(0,2,1)  # (N, H*W, C)
            else:
                query = neighbor_feature_flat[0:1,...].permute(0,2,1)  # (1, H*W, C)
                key = neighbor_feature_flat.permute(0,2,1)  # (N, H*W, C)
            
            value = neighbor_feature_flat.permute(0,2,1)

            if self.with_scm:
                confidence_map = batch_confidence_map[b]
                fused_feature = self.encode_layer(query, key, value, confidence_map)  # (1, H*W, C)
            else:
                fused_feature = self.encode_layer(query, key, value)  # (1, H*W, C)
            
            fused_feature = fused_feature.permute(0,2,1).reshape(1, C, H, W)

            x_fuse.append(fused_feature)
        x_fuse = torch.concat(x_fuse, dim=0)
        return x_fuse

class MultiTransformerFusion(nn.Module):
    def __init__(self, index):
        l=1   
    def forward(self, index):
        return 0
 
class MultiFusion(nn.Module):
    def __init__(self, args, confidence_dim):
        super(MultiFusion, self).__init__()
        self.lidar_range = args['multi_args']['communication']['lidar_range']
        self.voxel_size_x, self.voxel_size_y, self.voxel_size_z = args['multi_args']['communication']['voxel_size']
        self.feature_stride = args['multi_args']['communication']['feature_stride']
        
        self.layer_nums = len(args['multi_args']['layer_nums'])
        self.discrete_ratio = args['multi_args']['communication']['voxel_size'][0]
        self.downsample_rate = args['multi_args']['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        self.get_mask = Communication(args['multi_args']['communication'], confidence_dim)
        self.pre_processor = build_preprocessor(args['multi_args']['communication']['preprocess'], train=args['train'])
        layer_nums = args['multi_args']['layer_nums']
        num_filters = args['multi_args']['num_filters']
        self.voxel_x, self.voxel_y, self.voxel_z = args['multi_args']['communication']['voxel_size']
        self.num_levels = len(layer_nums)
        self.max_cav = args['max_cav']
        self.transformer_flag = args['transformer_flag']
        
        self.lidar_range = args['lidar_range']
        self.bbx_args = args['postprocess']['anchor_args']
        self.left_hand = args['left_hand']
            
        self.transformer_net = RPN_transformer_deformable_mtf(128*3, args)
        self.bbx_net = BoxExcrator(in_channels=7, out_channels=128*3)
        self.bbx_project = nn.Linear(384, 384)
        self.redundancy_att = nn.ModuleList()

        for idx in range(self.num_levels):
            redunancy = RedundancyAttention(num_filters[idx], args['multi_args']['redundancyattention'])
            self.redundancy_att.append(redunancy)

    def regroup(self, confidence_map, cav_nums):
        cum_sum_len = torch.cumsum(cav_nums, dim=0)
        split_map = torch.tensor_split(confidence_map, cum_sum_len[:-1].cpu())
        return split_map
    
    def forward(self, score_single, bbxs_single_corner, bbxs_single, confidence_map, x, cav_nums, pairwise_t_matrix, tau, bbx_id, backbone):
        
        _, C, H, W = x.shape
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2
        batch_size = len(cav_nums)
        ups = []
        with_resnet = True if hasattr(backbone, 'resnet') else False
        if with_resnet:
            feats = backbone.resnet(x)
        for i in range(self.layer_nums):
            x=feats[i] if with_resnet else backbone.blocks[i](x) 
               
            if i == 0:
                batch_feat = self.regroup(x, cav_nums)
                batch_confidence_map = self.regroup(confidence_map, cav_nums)
                mask_feature, mask_proposal, comm_feature = self.get_mask(batch_confidence_map, cav_nums, tau)
                selected_bbx_corner, selected_bbx, bbx_num, selected_score, selected_id = self.select_bbx(
                    bbxs_single_corner, mask_proposal, bbxs_single, score_single, bbx_id)                                                        
                selected_feature_list = []
                for b in range(batch_size):
                    selected_feature = mask_feature[b] * batch_feat[b]
                    final_feature = selected_feature
                    selected_feature_list.append(final_feature)
                x = torch.cat(selected_feature_list,dim=0)  

            batch_node_features = self.regroup(x, cav_nums)
            x_fuse = []
            for b in range(batch_size):
                N = cav_nums[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                C, H, W = batch_node_features[b].shape[1:]  # 64，100，252
                neighbor_feature = warp_affine_simple(batch_node_features[b],
                                                    t_matrix[0, :, :, :],
                                                    (H, W))                      
                x_fuse.append(self.redundancy_att[i](neighbor_feature))
            x_fuse = torch.stack(x_fuse)

            if len(backbone.deblocks) > 0:
                ups.append(backbone.deblocks[i](x_fuse))
            else:
                ups.append(x_fuse)

        if len(ups) > 1:
            x_fuse = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x_fuse = ups[0]

        if len(backbone.deblocks) > self.num_levels:
             x_fuse = backbone.deblocks[-1](x_fuse)

        if self.transformer_flag:
            bbx_offset_list = []
            bbx_score_list = []
            for b in range(batch_size):

                if selected_bbx[b].shape[0]==0:
                    selected_bbx[b], selected_score[b], selected_bbx_corner[b], selected_id[b] = self.generate_random_bbx(x_fuse)   

                bbx_feature = self.bbx_net(selected_bbx[b]) 
                _, bbx_offset, bbx_score = self.transformer_net(x_fuse[b], bbx_feature, selected_bbx[b], self.left_hand, selected_score[b], selected_id[b])

                bbx_offset_list.append(bbx_offset)
                bbx_score_list.append(bbx_score)
    
            return x_fuse, selected_bbx_corner, bbx_num, comm_feature, bbx_offset_list, selected_bbx, selected_id, bbx_score_list
        else:
            return x_fuse, selected_bbx_corner, bbx_num, comm_feature, selected_bbx
        
    def select_bbx(self, bbx_single_corner, mask_proposal, bbx_single, score_single, bbx_id):

        empty_list = torch.zeros((0,7), device=mask_proposal[0].device)
        empty_list_corner = torch.zeros((0,8,3), device=mask_proposal[0].device)
        empty_score = torch.zeros(0, device=mask_proposal[0].device)     
        selected_bbx_corner_batch = []
        selected_bbx_batch = []
        selected_score_batch = []
        selected_bbx_id_batch = []
        bbx_num = 0
        for b in range(len(mask_proposal)):
            selected_bbx_corner_list = []
            selected_bbx_list = []
            selected_score_list = [] 
            selected_bbx_id_list = []           
            for n in range(len(bbx_single_corner[b])):
                single_cav_corner_bbx = bbx_single_corner[b][n].to(torch.float64)
                single_cav_bbx = bbx_single[b][n] 
                single_cav_score = score_single[b][n]
                single_bbx_id = bbx_id[b][n]
                if len(single_cav_bbx) == 0:
                    selected_bbx_corner_list.append(empty_list_corner)
                    selected_bbx_list.append(empty_list)
                    selected_score_list.append(empty_score)
                    selected_bbx_id_list.append(empty_score)
                    continue
                center_point = (single_cav_corner_bbx[:, 0, :] + single_cav_corner_bbx[:, 6, :])/2
                center_point, mask = self.mask_points_by_range_v1(center_point, self.lidar_range)
                single_cav_corner_bbx = single_cav_corner_bbx[mask]
                single_cav_bbx = single_cav_bbx[mask]
                single_cav_score = single_cav_score[mask]
                single_bbx_id = single_bbx_id[mask]
                bev_bbx = torch.zeros(center_point.shape[0], 2, device=center_point.device)
                
                if self.left_hand:
                    bev_bbx[:, 0] = torch.div(center_point[:, 0] + self.lidar_range[3], self.voxel_x * self.feature_stride, rounding_mode='trunc')
                    bev_bbx[:, 1] = torch.div(center_point[:, 1] + self.lidar_range[4], self.voxel_y * self.feature_stride, rounding_mode='trunc')
                else:
                    bev_bbx[:, 0] = torch.div(center_point[:, 0] + self.lidar_range[3], self.voxel_x * self.feature_stride, rounding_mode='trunc')
                    bev_bbx[:, 1] = torch.div(self.lidar_range[4] - center_point[:, 1], self.voxel_y * self.feature_stride, rounding_mode='trunc')
                
                mask_proposal_i = mask_proposal[b][n].squeeze()
                y_indices = bev_bbx[:, 0].to(torch.int64)
                x_indices = bev_bbx[:, 1].to(torch.int64)
                mask_value = mask_proposal_i[x_indices, y_indices]
                mask_value = mask_value.bool()
                selected_bbx_corner = single_cav_corner_bbx[mask_value]
                selected_bbx = single_cav_bbx[mask_value]
                selected_score = single_cav_score[mask_value]
                selected_bbx_id = single_bbx_id[mask_value]
                selected_bbx_corner_list.append(selected_bbx_corner)
                selected_bbx_list.append(selected_bbx)
                selected_score_list.append(selected_score)
                selected_bbx_id_list.append(selected_bbx_id)
                if n==0:
                    bbx_num = bbx_num + single_cav_corner_bbx.shape[0]

            selected_score_batch.append(torch.cat(selected_score_list))
            selected_bbx_corner_batch.append(torch.cat(selected_bbx_corner_list))
            selected_bbx_batch.append(torch.cat(selected_bbx_list))
            selected_bbx_id_batch.append(torch.cat(selected_bbx_id_list))
            
        bbx_num_mean = bbx_num / len(mask_proposal)

        return selected_bbx_corner_batch, selected_bbx_batch, bbx_num_mean, selected_score_batch, selected_bbx_id_batch    

    def mask_points_by_range_v1(self, points, limit_range):
        """
        Remove the lidar points out of the boundary.

        Parameters
        ----------
        points : np.ndarray
            Lidar points under lidar sensor coordinate system.

        limit_range : list
            [x_min, y_min, z_min, x_max, y_max, z_max]

        Returns
        -------
        points : np.ndarray
            Filtered lidar points.
        """

        mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3])\
               & (points[:, 1] > limit_range[1]) & (
                       points[:, 1] < limit_range[4]) \
               & (points[:, 2] > limit_range[2]) & (
                   points[:, 2] < limit_range[5])

        points = points[mask]

        return points, mask

    def generate_random_bbx(self, x_fuse):
        """
        return:
        selected_bbx (n,7)
        """
        B, C, H, W = x_fuse.shape
        x_min, y_min, z_min, x_max, y_max, z_max = self.lidar_range
        h, w, l, r = self.bbx_args['h'], self.bbx_args['w'], self.bbx_args['l'], self.bbx_args['r']
        x = torch.FloatTensor(1).uniform_(x_min, x_max)
        y = torch.FloatTensor(1).uniform_(y_min, y_max)
        z = torch.FloatTensor(1).uniform_(z_min, z_max)
        r = torch.FloatTensor(1).uniform_(r[0], r[1])
        bbx = torch.tensor([x,y,z,h,w,l,r], device=x_fuse.device).unsqueeze(0)
        dx, dy, dz = l/2, w/2, h/2
        corners = torch.tensor([
            [x - dx, y - dy, z - dz],
            [x + dx, y - dy, z - dz],
            [x - dx, y + dy, z - dz],
            [x + dx, y + dy, z - dz],
            [x - dx, y - dy, z + dz],
            [x + dx, y - dy, z + dz],
            [x - dx, y + dy, z + dz],
            [x + dx, y + dy, z + dz],
        ], device=x_fuse.device).unsqueeze(0)
        score = torch.tensor([0.5], device=x_fuse.device)
        bbx_id = torch.randint(0, H*W*2, (1,))
        return bbx, score, corners, bbx_id
    