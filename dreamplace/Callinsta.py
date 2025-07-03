from thirdparty.INSTA.src.core.insta import INSTA
import torch



# def call_insta (pos, pin_offset_x, pin_offset_y, pin2node_map, num_cells, insta_path, design_path, input_path):
#     insta = INSTA()
#     path=insta.do_set_insta_path(path=insta_path, design_name=design_path, input_folderName=input_path)
#     print(path)
#     success = insta.do_initialization()
#     if not success:
#         raise RuntimeError("INSTA initialization failed")
#     # 提取 cell 位置坐标
#     node_x = pos[:num_cells]
#     node_y = pos[num_cells:]
#     # 保证设备一致
#     pin2node_map = pin2node_map.to(node_x.device)
#     pin_offset_x = pin_offset_x.to(node_x.device)
#     pin_offset_y = pin_offset_y.to(node_y.device)
#     # 计算 pin 坐标
#     pin_x = node_x[pin2node_map] + pin_offset_x
#     pin_y = node_y[pin2node_map] + pin_offset_y
#     # 将 pin 坐标传入 INSTA
#     insta.pin_pos_x = pin_x
#     insta.pin_pos_y = pin_y
#     # 更新 collateral 并执行传播
#     insta._precompute_collaterals()  # 更新每条 arc 的 delay
#     insta.do_diff_propagation(topk=1)  # 可微分传播
#     # 提取每个 arc 或 output pin 的 gradient
#     insta.do_extract_arc_grads()
#     slack = insta.slacks.to(node_x.device)  # 确保在同一个 device 上
#     # 示例：将某些时序指标当作 loss
#     slack_loss = torch.sum(torch.clamp(-slack, min=0.0))  # 负 slack 越大 loss 越高
#     return slack, slack_loss


class Call_Insta:
    def __init__(self, insta_path, design_path, input_path, pin_offset_x, pin_offset_y, pin2node_map):
        self.insta = INSTA()
        self.insta.do_set_insta_path(path=insta_path, design_name=design_path, input_folderName=input_path)
        success = self.insta.do_initialization()
        if not success:
            raise RuntimeError("INSTA initialization failed")
        self.pin_offset_x = pin_offset_x
        self.pin_offset_y = pin_offset_y
        self.pin2node_map = pin2node_map

    def timing_loss(self, pos):
        if pos.dim() == 1:
         pos = pos.view(-1, 2) 
        pin_x = pos[self.pin2node_map, 0] + self.pin_offset_x
        pin_y = pos[self.pin2node_map, 1] + self.pin_offset_y
        pin_coords = torch.stack([pin_x, pin_y], dim=1)
        self.insta.timing_tensors['pin_coords'] = pin_coords
        self.insta._precompute_collaterals()
        self.insta.do_diff_propagation()
        endpoint_gids = self.insta.timing_tensors['dest_node_tensor']
        slack = self.insta.timing_tensors['Gid_2_slack'][endpoint_gids]
        slack_loss = torch.sum(torch.clamp(-slack, min=0.0))
        return slack_loss