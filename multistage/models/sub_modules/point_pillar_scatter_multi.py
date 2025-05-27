import torch
import torch.nn as nn


class PointPillarScatterMulti(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']
        assert self.nz == 1

    def forward(self, data_dict):
        pillar_features, coords = data_dict['pillar_features'], data_dict[
            'voxel_coords']

        spatial_feature = torch.zeros(
            self.num_bev_features,
            self.nz * self.nx * self.ny,
            dtype=pillar_features.dtype,
            device=pillar_features.device)

        indices = coords[:, -3] + \
                    coords[:, -2] * self.nx + \
                    coords[:, -1]
        indices = indices.type(torch.long)

        pillars = pillar_features.t()
        spatial_feature[:, indices] = pillars

        spatial_feature = spatial_feature.view(self.num_bev_features * self.nz, self.ny, self.nx)

        data_dict['spatial_features'] = spatial_feature

        return data_dict

