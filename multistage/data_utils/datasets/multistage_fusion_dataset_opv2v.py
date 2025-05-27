
import math
import random

import numpy as np
import torch

import multistage.data_utils.datasets
from multistage.data_utils.pre_processor import build_preprocessor
from multistage.data_utils.post_processor import build_postprocessor
from multistage.data_utils.datasets import basedataset
from collections import OrderedDict
from multistage.utils.transformation_utils import x1_to_x2
from multistage.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from multistage.utils import box_utils


class MultistageFusionDatasetOpv2v(basedataset.BaseDataset):
    def __init__(self, params, visualize, train=True):
        super(MultistageFusionDatasetOpv2v, self).__init__(params, visualize, train)
        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first = True
        if 'proj_first' in params['fusion']['args'] and \
            not params['fusion']['args']['proj_first']:
            self.proj_first = False

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        self.cur_ego_pose_flag = True if 'cur_ego_pose_flag' not in \
            params['fusion']['args'] else \
            params['fusion']['args']['cur_ego_pose_flag']
        
        self.pre_processor = build_preprocessor(params['preprocess'], train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

    def __getitem__(self, index):
        base_data_dict = self.retrieve_base_data(index)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []
        
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0
      
        pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.max_cav)
        
        object_stack = []
        object_id_stack = []
        processed_voxel_stack = []
        processed_origin_lidar_dict = []
        spatial_correction_matrix = []
        origin_lidar_nums = []

        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > multistage.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)

            processed_voxel_stack.append(selected_cav_processed['processed_voxel'])
            processed_origin_lidar_dict.append(selected_cav_processed['origin_lidar'])
            origin_lidar_nums.append(selected_cav_processed['origin_lidar'].shape[0])
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            spatial_correction_matrix.append(
                selected_cav_base['params']['spatial_correction_matrix'])
        

        cav_nums = len(processed_voxel_stack)
        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack) 
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'] * cav_nums, 7))
        mask = np.zeros(self.params['postprocess']['max_num'] * cav_nums)
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        
        processed_voxel = self.merge_features_to_dict(processed_voxel_stack)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center,
                                                        anchors=anchor_box,
                                                        mask=mask)
                                                        
        processed_data_dict['ego'].update({'label_dict': label_dict,
                                            'anchor_box': anchor_box,
                                            'object_bbx_center': object_bbx_center,
                                            'object_bbx_mask': mask,
                                            'object_ids': [object_id_stack[i] for i in unique_indices],
                                            'processed_voxel': processed_voxel,
                                            'origin_lidar': processed_origin_lidar_dict,
                                            'cav_nums': cav_nums,
                                            'origin_lidar_nums': origin_lidar_nums, 
                                            'pairwise_t_matrix_feature': pairwise_t_matrix})
        
        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_lidar_pose):

        selected_cav_processed = {}

        transformation_matrix = selected_cav_base['params']['transformation_matrix'] 

        lidar_np = selected_cav_base['lidar_np']

        lidar_np = shuffle_points(lidar_np)

        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)

        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                                   transformation_matrix)

        lidar_np = mask_points_by_range(lidar_np,
                                            self.params['preprocess']['cav_lidar_range'])

        # generate the bounding box(n, 7) under the cav's space
        # object_bbx_center 7:x,y,z,h,w,l,yaw
        object_bbx_center, object_bbx_mask, object_ids = \
                self.post_processor.generate_object_center([selected_cav_base],
                                                            ego_lidar_pose)
        
        voxel_dict = self.pre_processor.preprocess(lidar_np)

        lidar_np = torch.tensor(lidar_np)

        selected_cav_processed.update({'processed_voxel': voxel_dict})
        selected_cav_processed.update({'origin_lidar': lidar_np})
        selected_cav_processed.update({'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
                                       'object_bbx_mask': object_bbx_mask,
                                       'object_ids': object_ids
                                       })

        return selected_cav_processed
    
    def collate_batch_train(self, batch):
        
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_voxel_list = []
        
        label_dict_list = []
        cav_nums = []
        origin_lidar_list = []
        origin_lidar_num_list = []
        pairwise_t_matrix_feature_list = []
        object_ids = []
        anchor_box_list = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            ego_dict['object_bbx_center'] = torch.tensor(ego_dict['object_bbx_center'])
            object_bbx_center.append(ego_dict['object_bbx_center'])
            ego_dict['object_bbx_mask'] = torch.tensor(ego_dict['object_bbx_mask'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_voxel_list.append(ego_dict['processed_voxel'])
            object_ids.append(ego_dict['object_ids'])
            origin_lidar_list.append((ego_dict['origin_lidar']))
            label_dict_list.append(ego_dict['label_dict'])
            cav_nums.append(ego_dict['cav_nums'])
            origin_lidar_num_list.append(ego_dict['origin_lidar_nums'])
            pairwise_t_matrix_feature_list.append(ego_dict['pairwise_t_matrix_feature'])
            anchor_box_list.append(ego_dict['anchor_box'])

        pairwise_t_matrix_feature = torch.from_numpy(np.array(pairwise_t_matrix_feature_list))
        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.cat(object_bbx_center)
        object_bbx_mask = torch.cat(object_bbx_mask)
        cav_nums = torch.from_numpy(np.array(cav_nums))
        anchor_box = torch.from_numpy(np.array(anchor_box_list))

        processed_voxel_dict = self.merge_features_to_dict(processed_voxel_list)

        processed_voxel_torch_dict = \
             self.pre_processor.collate_batch(processed_voxel_dict)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_voxel_torch_dict,
                                   'object_ids': object_ids[0],
                                   'origin_lidar': origin_lidar_list,
                                   'label_dict': label_torch_dict, 
                                   'record_len':cav_nums,
                                   'anchor_box_list': anchor_box, 
                                   'origin_lidar_nums':origin_lidar_num_list, 
                                   'pairwise_t_matrix': pairwise_t_matrix_feature
                                   })

        return output_dict
    
            
    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict              

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        # if lidar projected to ego first, then the pairwise matrix
        # becomes identity
        pairwise_t_matrix[:, :] = np.identity(4)

        return pairwise_t_matrix
    
    def get_matrix(self, base_data_dict, max_cav):

        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        t_list = []
        # save all transformation matrix in a list in order first.
        for cav_id, cav_content in base_data_dict.items():
            t_list.append(cav_content['params']['transformation_matrix'])

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    t_matrix = np.eye(4)
                    pairwise_t_matrix[i, j] = t_matrix
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix

    def get_unique_label(self, object_stack, object_id_stack):
        # IoU
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))  # [100,7]
        mask = np.zeros(self.params['postprocess']['max_num'])  # [100,]
        if len(object_stack) > 0:
            # exclude all repetitive objects    
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack) if len(object_stack) > 1 else object_stack[0]
            object_stack = object_stack[unique_indices]
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            updated_object_id_stack = [object_id_stack[i] for i in unique_indices]
        else:
            updated_object_id_stack = object_id_stack
        return object_bbx_center, mask, updated_object_id_stack


    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego'][
                        'anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch})

        return output_dict


    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process_multitest_opv2v(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor                            
        

        







    



