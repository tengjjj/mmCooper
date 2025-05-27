
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from multistage.models.sub_modules.pillar_vfe import PillarVFE
from multistage.models.sub_modules.point_pillar_scatter import PointPillarScatter
from multistage.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from multistage.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from multistage.models.fuse_modules.multi_fusion import MultiFusion
from multistage.data_utils.post_processor import build_postprocessor
from multistage.data_utils.pre_processor import build_preprocessor
from multistage.utils import box_utils
from multistage.models.sub_modules.downsample_conv import DownsampleConv

class PointPillarMulti(nn.Module):
    def __init__(self, args):
        super(PointPillarMulti, self).__init__()

        self.target_args_score = args['postprocess']['target_args']['score_threshold']
        self.nms_thresh = args['postprocess']['nms_thresh']
        self.transformer_flag = args['transformer_flag']
        self.confidence_dim = 2

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.fuse = MultiFusion(args, self.confidence_dim)

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        
        self.confidence_generate = nn.Conv2d(128 * 2, self.confidence_dim, kernel_size=3, padding=1)
                
        self.post_processor = build_postprocessor(args['postprocess'], train=args['train'])
        self.order = args['postprocess']['order']
        self.lidar_range = args['lidar_range']
        self.voxel_x, self.voxel_y, self.voxel_z = args['voxel_size']
        
        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
        """

        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.fuse.parameters():
            p.requires_grad = False

        for p in self.confidence_generate.parameters():
            p.requires_grad = False

    def regroup(self, confidence_map, cav_nums):
        cum_sum_len = torch.cumsum(cav_nums, dim=0)
        split_map = torch.tensor_split(confidence_map, cum_sum_len[:-1].cpu())
        return split_map

    def forward(self, data_dict, tau):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']        
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        cav_nums = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'cav_nums':cav_nums}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        assert not torch.isnan(psm_single).any() or torch.isnan(rm_single).any()

        bbxs_single_corner, scores_single, bbxs_single, bbx_id = self.generate_bbx(psm_single, rm_single, cav_nums, pairwise_t_matrix, data_dict['anchor_box_list'])
        
        confidence = self.confidence_generate(spatial_features_2d)
        
        if self.transformer_flag:
            fused_feature, selected_bbx_corner, bbx_num, comm_feature, bbx_offset, selected_bbx, selected_id, bbx_score = self.fuse(
                                                                            scores_single,
                                                                            bbxs_single_corner, 
                                                                            bbxs_single, 
                                                                            confidence, 
                                                                            batch_dict['spatial_features'], 
                                                                            cav_nums, 
                                                                            pairwise_t_matrix, 
                                                                            tau, 
                                                                            bbx_id, 
                                                                            self.backbone)
        else:
            fused_feature, selected_bbx_corner, bbx_num, comm_feature, selected_bbx = self.fuse(
                                                                            scores_single,
                                                                            bbxs_single_corner, 
                                                                            bbxs_single, 
                                                                            confidence, 
                                                                            batch_dict['spatial_features'], 
                                                                            cav_nums, 
                                                                            pairwise_t_matrix, 
                                                                            tau,
                                                                            bbx_id, 
                                                                            self.backbone)           

        if self.transformer_flag:
            B, _, H, W = fused_feature.shape
            offs, scores = self.generate_offset_score(selected_bbx, data_dict['anchor_box_list'], selected_id, bbx_score) 
            final_bbx = []
            off_id = []
            for b in range(B):
                
                offsets = torch.zeros_like(offs)
                mask = torch.gt(scores[b], 0.01).squeeze(-1)
                bbx_mask = mask[selected_id[b].long()]
                selected_bbx_offset = bbx_offset[b]
                bbx_offset[b] = bbx_offset[b][bbx_mask]
                off_mask_id = selected_id[b][bbx_mask]
                off_id.append(off_mask_id)
                
                offsets[b, off_mask_id.long(), :] = offs[b, off_mask_id.long(), :]
                offsets[b, off_mask_id.long(), :] += bbx_offset[b]
                final_bbx.append(selected_bbx[b] + selected_bbx_offset)

            offsets = offsets.view(B, H, W, -1)
        
        total_comm = torch.log2((comm_feature * 64 + 7 * bbx_num) * 32 / 8)
        if torch.isinf(total_comm):
            total_comm = 0
        fused_feature = self.shrink_conv(fused_feature)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        if self.transformer_flag:
            output_dict = {'psm': psm,
                        'rm': rm,
                        'bbx': selected_bbx,
                        'bbx_id': selected_id,
                        'off_id': off_id,
                        'bbx_score': scores,  
                        'total_comm_rate':total_comm, 
                        'bbx_off': offsets,
                        'selected_bbx': selected_bbx_corner}
        else:
            output_dict = {'psm': psm,
                           'rm': rm,
                           'bbx': selected_bbx,
                           'total_comm_rate':total_comm
                            }
        return output_dict


    def generate_bbx(self, psm, rm, cav_nums, pairwise_t_matrix, anchor_box):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        empty_list = torch.tensor([]).to(psm)
        batch_psm = self.regroup(psm, cav_nums)
        batch_rm = self.regroup(rm, cav_nums)
        pred_box3d_corner_batch = []
        scores_batch = []
        pred_box3d_batch = []
        bbx_id_batch = []
        for b in range(len(cav_nums)):
            pred_box3d_corner_list = []
            scores_list = []
            pred_box3d_list = []
            bbx_id_list = []
            for n in range(cav_nums[b]):

                prob = torch.sigmoid(batch_psm[b][n:n+1, :, :, :].permute(0, 2, 3, 1))
                prob = prob.reshape(1, -1)

                batch_box3d = self.post_processor.delta_to_boxes3d(batch_rm[b][n:n+1, :, :, :], anchor_box[b])

                mask = torch.gt(prob, self.target_args_score)
                mask = mask.view(1, -1)
                mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
                bbx_id = torch.nonzero(mask)[..., 1]

                boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
                scores = torch.masked_select(prob[0], mask[0])
       
                boxes3d_corner = \
                        box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.order)

                projected_boxes3d = \
                        box_utils.project_box3d(boxes3d_corner, pairwise_t_matrix[b][0,n].type(torch.float32))

                projected_boxes2d = \
                        box_utils.corner_to_standup_box_torch(projected_boxes3d)

                boxes2d_score = \
                        torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                if len(boxes2d_score) ==0 or len(projected_boxes3d) == 0:
                    pred_box3d_corner_list.append(empty_list)
                    scores_list.append(empty_list)
                    pred_box3d_list.append(empty_list)
                    bbx_id_list.append(empty_list)
                    continue

                scores = boxes2d_score[:, -1]
                keep_index_1 = box_utils.remove_large_pred_bbx(projected_boxes3d)
                keep_index_2 = box_utils.remove_bbx_abnormal_z(projected_boxes3d)
                keep_index = torch.logical_and(keep_index_1, keep_index_2)
                bbx_id = bbx_id[keep_index]
                pred_box3d_corner = projected_boxes3d[keep_index]
                scores = scores[keep_index]
                pred_box3d = boxes3d[keep_index]

                pred_box3d_corner_list.append(pred_box3d_corner)
                scores_list.append(scores)
                pred_box3d_list.append(pred_box3d)
                bbx_id_list.append(bbx_id)

            pred_box3d_corner_batch.append(pred_box3d_corner_list)
            scores_batch.append(scores_list)
            pred_box3d_batch.append(pred_box3d_list)
            bbx_id_batch.append(bbx_id_list)

        return pred_box3d_corner_batch, scores_batch, pred_box3d_batch, bbx_id_batch
    
    def generate_offset_score(self, bbx, anchors, bbx_id, bbx_score):

        B, H, W, _, _ = anchors.shape
        off_list = []
        score_list = []
        for b in range(len(bbx)):
            off = torch.zeros((H*W*2, 7), device=anchors.device)
            score = torch.full((H*W*2, 1), -1e8, device=anchors.device)
            
            anchors_reshaped = anchors[b].view(-1, 7).float()
            anchors_d = torch.sqrt(
                anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
            anchors_d = anchors_d.repeat(2, 1).transpose(0, 1)
            bbx_single = bbx[b]
            id_single = bbx_id[b].long()
            score_single = bbx_score[b]

            score[id_single] = score_single
            bbx_reshaped = anchors_reshaped.clone()
            bbx_reshaped[id_single] = bbx_single
            off[..., [0, 1]] = (bbx_reshaped[..., [0, 1]] - anchors_reshaped[..., [0, 1]]) / anchors_d
            off[..., [2]] = (bbx_reshaped[..., [2]] - anchors_reshaped[..., [2]]) / anchors_reshaped[..., [3]]
            off[..., [3, 4, 5]] = torch.log((bbx_reshaped[..., [3, 4, 5]]) / anchors_reshaped[..., [3, 4, 5]])
            off[..., 6] = bbx_reshaped[..., 6] - anchors_reshaped[..., 6]
            off_list.append(off)
            score_list.append(score)
            assert not torch.isnan(off).any()

        offs = torch.stack(off_list)
        scores = torch.stack(score_list)
        
        return offs, scores
    

