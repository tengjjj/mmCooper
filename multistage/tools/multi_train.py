import os
import torch
import torch.distributed.launch


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

nproc_per_node = 2
use_env = True
# config_file = "/home/tengjian/code/multistage/multi/hypes_yaml/point_pillar_where2comm_opv2v.yaml"
# checkpoint_folder = "your_checkpoint_folder"

torch.distributed.launch.main([
    '--nproc_per_node', str(nproc_per_node),
    '--use_env' if use_env else '',
    'multi/tools/train.py',
    # '--hypes_yaml', config_file,
    # '--model_dir', checkpoint_folder if checkpoint_folder else ''
])
