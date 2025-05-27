import torch
import math

def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points 

def filtering(idx, voxel_num, cluster_num, filter_ratio, device):
    weights = []

    for i in range(cluster_num):
        cluster_index = torch.where(idx == i)[0].to(device)
        weight = torch.sum(voxel_num[cluster_index]).reshape(1, 1)
        # weight = torch.sum(voxel_num[cluster_index]).reshape(1, 1) / cluster_index.shape[0]
        weights.append(weight)
    weights = torch.cat(weights)
    top_center = torch.topk(weights, k=math.ceil(cluster_num * filter_ratio), dim=0)[1]
    mask = torch.isin(idx, top_center)
    
    return mask
        

def dpc_knn(x, cluster_num, device, filter_ratio = 0.1, k=0.02):
    """
    x: batch_dict
    cluster_num: number of clusters
    k: truncation distance
    """

    previous_points_num = 0

    voxel_coords = x['voxel_coords']
    N = torch.max(voxel_coords[:, 0])
    filtered_masks = []
    ego_voxel = voxel_coords[voxel_coords[:, 0] == 0]
    ego_mask = torch.ones(ego_voxel.shape[0], dtype=torch.bool)
    filtered_masks.append(ego_mask)
    previous_points_num = previous_points_num + ego_voxel.shape[0]
    
    if N == 0:
        return ego_mask, 0.0
    else:
        for i in range(1, N + 1):
            single_cav_voxel = voxel_coords[voxel_coords[:, 0] == i].float()
            n = single_cav_voxel.shape[0]
            if single_cav_voxel.shape[0] == 0:
                continue
            elif single_cav_voxel.shape[0] < cluster_num:
                mask = torch.ones(single_cav_voxel.shape[0], dtype=torch.bool)
                filtered_masks.append(mask)
                previous_points_num = previous_points_num + n
            else:
                dist_matrix = torch.cdist(single_cav_voxel, single_cav_voxel, p=2) / (n ** 0.5)
                dist_nearest, index_nearest = torch.topk(dist_matrix, k=int(round(k * n)), dim=-1, largest=False)
                density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
                density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
                mask = density[None, :] > density[:, None]
                mask = mask.type(voxel_coords.dtype)
                dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None]
                dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
                score = dist * density
                _, index_down = torch.topk(score, k=cluster_num, dim=-1)
                dist_matrix = index_points(dist_matrix.unsqueeze(0), index_down.unsqueeze(0))
                idx_cluster = dist_matrix.squeeze(0).argmin(dim=0)
                filtered_mask = filtering(idx_cluster, x['voxel_num_points'][previous_points_num:previous_points_num + n], cluster_num, filter_ratio, device)
                filtered_masks.append(filtered_mask)
                previous_points_num = previous_points_num + n

        filtered_masks = [mask.to(device) for mask in filtered_masks]
        filtered_masks = torch.cat(filtered_masks)
        n = int(filtered_masks.sum()) - ego_voxel.shape[0]
        com_volum = torch.sum(x['voxel_num_points'][filtered_masks][ego_voxel.shape[0]:]) * 4 * 4
        com_volum += n * 4 * 4
        com_volum += n * 4
        com_volum = com_volum / N
        
        return filtered_masks, com_volum
