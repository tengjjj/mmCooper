import torch

input = torch.rand(1,64,100,352)

bbx_feat_query = torch.rand(50,64)

position = torch.randint(0, 100, (50,2))

position_embeding = torch.nn.Linear(2,64)
offset_emd = torch.nn.Linear(64,2)
scores_emd = torch.nn.Linear(64,1)
value_emd = torch.nn.Linear(64,64)

p_out = position_embeding(position.to(torch.float32))

input_with_pos = input[:, :, position[:,0], position[:, 1]].view(1,64,-1).squeeze(0).transpose(1,0) + p_out

offsets = offset_emd(bbx_feat_query)

final_pos = offsets + position

score = torch.softmax(scores_emd(bbx_feat_query), dim=-1)

value = value_emd(input_with_pos)



