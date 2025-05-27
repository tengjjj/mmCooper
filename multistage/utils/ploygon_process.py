import torch
from shapely.geometry import Polygon
import torchvision.ops as ops

def transform_point(boxes2d, tx, ty, left_hand):
    boxes = boxes2d.clone()
    if left_hand:
        boxes[:, :, 0] = boxes[:, :, 0] + tx
        boxes[:, :, 1] = boxes[:, :, 1] + ty
    else:
        boxes[:, :, 0] = boxes[:, :, 0] + tx
        boxes[:, :, 1] = ty - boxes[:, :, 1]
    # boxes = boxes[:, :, [1, 0]]

    return boxes

def grid_scatter(boxes, grid_rows, grid_cols, grid_size, bbx_feature, feature_stride):

    C = bbx_feature.shape[1]
    grid_features = torch.zeros(C, grid_rows, grid_cols, device=boxes.device)
    grid_counts = torch.zeros(grid_rows, grid_cols, device=boxes.device)
#     x_min, _ = boxes[..., 0].min(dim=1)
#     x_max, _ = boxes[..., 0].max(dim=1)
#     y_min, _ = boxes[..., 1].min(dim=1)
#     y_max, _ = boxes[..., 1].max(dim=1)

#     # x_min_idx = (x_min // (stride * grid_size))
#     # x_max_idx = (x_max // (stride * grid_size))
#     # y_min_idx = (y_min // (stride * grid_size))
#     # y_max_idx = (y_max // (stride * grid_size))
    for box, feature in zip(boxes, bbx_feature):

        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]
        grid_x_min = torch.div(min(x1,x2,x3,x4), (grid_size * feature_stride), rounding_mode='trunc').int()
        grid_x_min = max(grid_x_min, 0)
        grid_x_max = torch.div(max(x1,x2,x3,x4), (grid_size * feature_stride), rounding_mode='trunc').int()
        grid_x_max = max(grid_x_max, 0)
        grid_y_min = torch.div(min(y1,y2,y3,y4), (grid_size * feature_stride), rounding_mode='trunc').int()
        grid_y_min = max(grid_y_min, 0)
        grid_y_max = torch.div(max(y1,y2,y3,y4), (grid_size * feature_stride), rounding_mode='trunc').int()
        grid_y_max = max(grid_y_max, 0)

        grid_features[:, grid_x_min:grid_x_max+1, grid_y_min:grid_y_max+1] += feature.view(C, 1, 1)
        grid_counts[grid_x_min:grid_x_max+1, grid_y_min:grid_y_max+1] += 1

    grid_counts[grid_counts == 0] = 1
    grid_features = grid_features / grid_counts

    assert not torch.all(grid_features==0)

    return grid_features

def grid_scatter_center_point(boxes, grid_rows, grid_cols, grid_size, bbx_feature, stride):
    center_point = (boxes[:, 0, :] + boxes[:, 2, :]) / 2

    center_grid_loc = center_point // (grid_size * stride)

    grid_feature = torch.zeros((grid_rows, grid_cols, bbx_feature.shape[1]), dtype=torch.float, device= boxes.device)
    counter = torch.zeros((grid_rows, grid_cols), dtype=torch.float32, device= boxes.device)

    x_coords = center_grid_loc[:, 0].long()
    y_coords = center_grid_loc[:, 1].long()
    grid_feature.index_put_((x_coords, y_coords), bbx_feature, accumulate=True)
    counter.index_put_((x_coords, y_coords), torch.ones(bbx_feature.shape[0], dtype=torch.float32, device= boxes.device), accumulate=True)
    counter[counter == 0] = 1

    grid_feature /= counter.unsqueeze(-1)
       
    return grid_feature.permute(2,0,1)
