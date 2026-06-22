
import torch
import torch.nn as nn
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template_gaussian import RoIHeadTemplateGaussian

class VoxelRCNNHeadGaussian(VoxelRCNNHead, RoIHeadTemplateGaussian):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        # Initialize VoxelRCNNHead but we want RoIHeadTemplateGaussian methods to take precedence
        # Since VoxelRCNNHead inherits from RoIHeadTemplate, and we inherit from both...
        # We need to be careful.
        # Actually, if we just inherit VoxelRCNNHead, we get RoIHeadTemplate methods.
        # If we want to override generate_predicted_boxes, we can just define it here or mixin.
        # But for clarity based on "create a copy", let's reimplement forward to use the new method.

        super().__init__(backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class, **kwargs)

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI Grid Pool
        roi_grid_pool_feats = []
        for pl in self.roi_grid_pool_layers:
            roi_grid_pool_feats.append(pl(batch_dict))

        roi_grid_pool_feats = torch.cat(roi_grid_pool_feats, dim=1)

        # Grid Attention (if exists in model code, VoxelRCNNHead usually has it but maybe not in this version)
        # Checking VoxelRCNNHead code read previously... it has shared_fc_list

        # Shared FC
        shared_features = roi_grid_pool_feats.view(roi_grid_pool_feats.shape[0], -1)
        for i in range(len(self.shared_fc_layer)):
            shared_features = self.shared_fc_layer[i](shared_features)

        # Heads
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds, batch_box_uncertainty = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'],
                rois=batch_dict['rois'],
                cls_preds=rcnn_cls,
                box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            # [Gaussian Mod]
            if batch_box_uncertainty is not None:
                batch_dict['batch_box_uncertainty'] = batch_box_uncertainty

            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
