from thirdparty.INSTA.src.core.insta import INSTA
import torch



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